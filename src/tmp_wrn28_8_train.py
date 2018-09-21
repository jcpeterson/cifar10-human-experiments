import numpy as np
import sklearn.metrics as metrics

# import wide_residual_network as wrn
#import models.wrn as wrn
from models.wrn_contrib import WideResidualNetwork as wrn

from keras.datasets import cifar10
import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

from keras import backend as K



batch_size = 100
nb_epoch = 100
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX = (trainX - trainX.mean(axis=0)) / (trainX.std(axis=0))
testX = testX.astype('float32')
testX = (testX - testX.mean(axis=0)) / (testX.std(axis=0))

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

#init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)
init_shape = (32, 32, 3)
#init_shape = (3, 32, 32)


model = wrn(depth=28, width=8, dropout_rate=0.0,
            include_top=True, weights='cifar10',
            input_tensor=None, input_shape=init_shape,
            classes=10, activation='softmax')

# model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=4, k=8, dropout=0.0)

# model.summary()

#model.load_weights("weights/wrn-28-8.h5")

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

print(model.evaluate(testX, testY))
exit()

# model.load_weights("weights/WRN-28-8 Weights.h5")
# print("Model loaded.")

model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size + 1, nb_epoch=nb_epoch,
                   callbacks=[callbacks.ModelCheckpoint("WRN-28-8 Weights.h5", monitor="val_acc", save_best_only=True)],
                   validation_data=(testX, testY),
                   validation_steps=testX.shape[0] // batch_size,)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yPred = kutils.to_categorical(yPred)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)


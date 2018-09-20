from __future__ import print_function

import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

from rutils import cifarH10_load_data



batch_size = 100
nb_classes = 10
#nb_epoch = 300
nb_epoch = 300

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, 
                          nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, 
                          dropout_rate=dropout_rate, weights=None)

model.summary()
# exit()
# optimizer = Adam(lr=1e-3) # Using Adam instead of SGD to speed up training
optimizer = Adam(lr=1e-4)

for layer in model.layers:
    layer.trainable = False
model.get_layer('dense_1').trainable = True
model.get_layer('batch_normalization_38').trainable = True
model.get_layer('batch_normalization_39').trainable = True
model.get_layer('conv2d_38').trainable = True
model.get_layer('conv2d_39').trainable = True

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=["accuracy"])

(trainX, trainY), (testX, testY) = cifar10.load_data()

# trainX = trainX.astype('float32')
# testX = testX.astype('float32')
# 
# trainX = densenet.preprocess_input(trainX)
# testX = densenet.preprocess_input(testX)
# 
# Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

X, y = cifarH10_load_data('', option='aggregated')
print('data loaded...')
print(X.shape, y.shape)

X = X.astype('float32')
X = densenet.preprocess_input(X)

X_human_train = X[0:8000].copy()
X_human_test = X[8000:].copy()

y_human_train = y[0:8000].copy()
y_human_test = y[8000:].copy()

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0)

# Load model
weights_file="DenseNet-40-12-CIFAR10.h5"
if os.path.exists(weights_file):
    model.load_weights(weights_file, by_name=True)
    print("Model loaded.")

model.summary()

out_dir="weights/"

lr_reducer      = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                    cooldown=0, patience=5, min_lr=1e-5)
model_checkpoint= ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                  save_weights_only=True, verbose=1)

#callbacks=[lr_reducer, model_checkpoint]
callbacks=[lr_reducer]

print('human loss before training:')
print(model.evaluate(X_human_test, y_human_test))

test_interval = 1
for _ in range(nb_epoch/test_interval):

    # yPreds = model.predict(X_human_test)
    # yPred = np.argmax(yPreds, axis=1)
    # yTrue = testY[8000:]
    # accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    # print("Label accuracy:", accuracy)

    print('Human:', model.evaluate(X_human_test, y_human_test))
    print('Label:', model.evaluate(X_human_test, Y_test[8000:]))
    
    model.fit_generator(generator.flow(X_human_train, y_human_train, batch_size=batch_size),
                        steps_per_epoch=len(X_human_train) // batch_size, epochs=test_interval,
                        callbacks=callbacks,
                        validation_data=(X_human_test, y_human_test),
                        validation_steps=X_human_test.shape[0] // batch_size, verbose=1)


import os, sys
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import rutils
import rdatagen

x_test, y_test, y_agg = rutils.return_dataset('aggregated')
x_test, y_test, y_sparse, x_idx = rutils.return_dataset('sparse')


datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1562,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1562,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False, # randomly flip images
            zoom_range=0.1) # zoom in a bit

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_test, seed = 9)

datagen_agg = rdatagen.datagen_aggregated(datagen, x_test, y_test,
                                          y_agg, 1000)
datagen_sp = rdatagen.datagen_sparse(datagen, x_test, x_idx, y_test,
                                          y_sparse, 1000, 50)

for items in datagen_sp:
    for arr in items:
        pass
#        try:
#            print(arr.shape)
#        except:
#            print([x.shape for x in arr])
print('finished')

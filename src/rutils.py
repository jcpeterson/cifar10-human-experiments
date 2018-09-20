
######################################### imports
import os
import sys

import numpy as np
from six.moves import cPickle

import cifarH10_load # new load functions module

from tensorflow.python.keras import backend as K #comment back in 

from tensorflow.python.keras.datasets.cifar10 import load_data

########################################## run
yourpath = os.getcwd() # this should give the right dir, but can be changed if necessary
d_path = os.path.abspath(os.path.join(yourpath, os.pardir, 'data'))
print('loading path, after joining: ', d_path)

def return_dataset(option='aggregated'):
    """Takes in an option for our data, 'aggregated' or 'sparse'
    and returns the image data, original labels, and data_specific
    labels and indices"""

    (x_train, y_train), (x_test, y_test) = load_data()

    x_train, y_train = None, None
    
    if option == 'aggregated':
        y_agg = cifarH10_load.load_aggregated(d_path)
        return x_test, y_test, y_agg

    elif option == 'sparse':
        y_sparse, x_idx = cifarH10_load.load_sparse(d_path)
        return x_test, y_test, y_sparse, x_idx

data = return_dataset('sparse')

print('print test')
print([x.shape for x in data])




# coding: utf-8

"""
This module load the cifarH10 data in as closely related format to the cifar10 test set data as possible. 

Input as path_to_data the parent directory of the cifarH10-data image directory. Options are explained below, including 'aggregated', 'sparse', and 'test_batch', which returns what the cifar10 functions called on the test_batch would have.


The normal load batch and data functions for the cifar library return the following:

(x_train, y_train), (x_test, y_test), where x's are each a matrix of rgb values (columns) for all test set images (rows) and y's are each 1-d numpy arrays of ground-truth labels (as integers).

We cannot quite keep this format, due to the structure of our guesses. The differences are given below:

In aggregated mode, x_whole is a matrix containing the rgb values (columns) for all cifar10 test set images (rows)---the same format as the cifar10 functions given above. y_whole is no longer a 1-d numpy array: now it is a 2-d numpy array, containing
number of guess per image (row) per category (column).

In sparse mode, y_whole is in the same format as the cifar10 functions given above. As the matrix of rgbs values (columns) for the images (rows) corresponding to the guesses in y_whole would be very large, the functions return the original cifar10 image matrix (as in cifar10) and a 1-d numpy array of indices for this matrix (giving the image corresponding to each guess; therefore, |guesses| long).
The full matrix can be built using the cifarH10_sparse_load_full_data(data, data_idx) function, with the possibility of batching guesses and data_idx first.

"""

import os
import sys

import numpy as np
from six.moves import cPickle

from tensorflow.python.keras import backend as K #comment back in 




def cifarH10_sparse_load_full_data(data, data_idx):
    """Takes in cifar10 test set matrix, and indices corresponding to 
    cifarH10 guesses. Creates LARGE (~12GB) matrix.
    Can batch indices and send in, as long as guesses batched equivalently"""
    return data[data_idx]

def cifarH10_load_batch(fpath, label_key = 'labels', option='aggregated'):
    """Internal utility for parsing CIFAR data.
    Arguments:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
           dictionary,
        option is aggregated or sparse or test_batch (original);
    Returns:
        A tuple `(data, labels)`, or `(data, data_idx, labels)`.
    """

    with open(fpath, 'rb') as f: # straight from keras
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    
    if option == 'sparse': # need idx to index into representation tensor
        data_idx = d['data_idx']
        
        return data, data_idx, labels

    else:
        return data, labels



def cifarH10_load_data(path_to_data, option = 'aggregated'):
    """Loads CIFARH10 dataset.
    Takes:
        Option: 'aggregated', 'sparse', or 'test_batch'; # incorporated into load data # test on both protocols
    Returns:
        Tuple of Numpy arrays: `(x_whole, y_whole)`.
        If option is 'aggregated', returns numpy matrix, where 
        rows are categories and columns are samples.
        Otherwise, returns sparse vector as keras function does.
    """
    dirname = 'cifarH10-data' # make sure to give to J in this format
  
    fpath = os.path.join(path_to_data, dirname, option)
  
    if option == 'sparse':
        x_whole, x_idx, y_whole = cifarH10_load_batch(fpath, 'labels', option)
        # in this case, labels are a list
        y_whole = np.reshape(y_whole, (len(y_whole), 1))

    elif option == 'aggregated': 
        # in this case labes are a numpy array (samples by categories)
        x_whole, y_whole = cifarH10_load_batch(fpath, 'labels', option)
        
        
    else:
        # in this case, labels are a list
        x_whole, y_whole = cifarH10_load_batch(fpath, 'labels', option)
        y_whole = np.reshape(y_whole, (len(y_whole), 1))
    
    if K.image_data_format() == 'channels_last':  
        x_whole = x_whole.transpose(0, 2, 3, 1)
    
    if option =='sparse':
        return (x_whole, x_idx, y_whole)
    
    else:
        return (x_whole, y_whole)


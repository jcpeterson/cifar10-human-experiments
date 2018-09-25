import os, sys
from six.moves import cPickle
import numpy as np

from keras.datasets.cifar10 import load_data as load_cifar10



def load_cifar10_plus_h(type='aggregate', 
	path='../data/cifar10h/'):

	"""
	   Type is 'aggregate' (probabilities) or 'sparse' 
	   (individual choices).

	   Returns the image data, original labels, and
	   human "labels".

	"""

	# load normal cifar10 data
	(X_train, y_train), (X_test, y_test) = load_cifar10()

	if type == 'aggregate':
	    y_agg = load_cifar10h_labels(type=type, path=path)
	    return X_train, y_train, X_test, y_test, y_agg

	elif type == 'sparse':
	    y_sparse, X_sparse_idx = \
	    	load_cifar10h_labels(type=type, path=path)
	    return X_train, y_train, X_test, y_test, \
	    	   y_sparse, X_sparse_idx

def load_cifar10h_labels(type='aggregate', 
	path='../data/cifar10h/'):

    """
       Type is 'aggregate' (probabilities) or 'sparse' 
       (individual choices).

    """

    if type == 'aggregate':
	    f_path = os.path.join(path, 'aggregate')
	    
	    with open(f_path, 'rb') as f: # straight from keras
	        if sys.version_info < (3,):
	            d = cPickle.load(f)
	        else:
	            d = cPickle.load(f, encoding='bytes')
	            # decode utf8
	            d_decoded = {}
	            for k, v in d.items():
	                d_decoded[k.decode('utf8')] = v
	            d = d_decoded
	    
	    labels = d['labels']
	    return labels

    elif type == 'sparse':
	    f_path = os.path.join(path, 'sparse')
	    
	    with open(f_path, 'rb') as f: # straight from keras
	        if sys.version_info < (3,):
	            d = cPickle.load(f)
	        else:
	            d = cPickle.load(f, encoding='bytes')
	            # decode utf8
	            d_decoded = {}
	            for k, v in d.items():
	                d_decoded[k.decode('utf8')] = v
	            d = d_decoded
	    
	    labels = d['labels']
	    labels = np.reshape(labels, (len(labels), 1))
	    data_idx = d['data_idx']
	    return labels, data_idx

def load_cifar10_1(version_string='v4', path='../data/cifar-10.1/'):

    """ 
        Load Ben Recht's CIFAR-10.1 data
        v4 is main dataset. v6 is appendix D in the paper

    """
    
    data_path = os.path.join(os.path.dirname(__file__), path)
    filename = 'cifar10.1'
    if version_string in ['v4', 'v6']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
        
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    labels = np.load(label_filepath)
    imagedata = np.load(imagedata_filepath)
    
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    # added to match keras cifar10 loading function output
    # goes from shape (n,) to (n,1)
    labels = labels.reshape(-1,1) #.astype(np.uint8) # dtype int32 to uint8
    
    return imagedata, labels




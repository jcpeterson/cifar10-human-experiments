import os
from six.moves import cPickle
import sys
import numpy as np

def load_aggregated(d_path):
    """Internal utility for parsing CIFAR data.
    Loads aggregated labels;
    Returns them.
    d_path is path to dict dir
    """
    f_path = os.path.join(d_path, 'aggregated')
    
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


def load_sparse(d_path):
    """Internal utility for parsing CIFAR data.
    Loads sparse labels and data row idx;
    Returns them.
    d_path is path to dict dir
    """
    f_path = os.path.join(d_path, 'sparse')
    
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

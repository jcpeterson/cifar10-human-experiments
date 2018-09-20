import os
import numpy as np

def load_cifar10_1(version_string='v4', path='../data/cifar-10.1/'):

    """ Load Ben Recht's CIFAR-10.1 data
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
    # print('Loading labels from file {}'.format(label_filepath))
    labels = np.load(label_filepath)
    # print('Loading image data from file {}'.format(imagedata_filepath))
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
    # goes from shape (n,) to (n,1) and dtype int32 to uint8
    labels = labels.reshape(-1,1).astype(np.uint8)
    
    return imagedata, labels




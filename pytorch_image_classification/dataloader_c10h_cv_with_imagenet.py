# coding: utf-8

""" 
    This version is post planning with Olga. We're adding a new test datset that
    consists of resized ImageNet images for cifar-like categories. We're also 
        adding cross-validation so our folders don't overlap as before.

    ImageNet image counts for each class:
        ('cat.npy', 3511)
        ('dog.npy', 9096)
        ('deer.npy', 2333)
        ('ship.npy', 8032)
        ('frog.npy', 8180)
        ('horse.npy', 1300)
        ('automobile.npy', 8204)
        ('airplane.npy', 6976)
        ('bird.npy', 8187)
        ('truck.npy', 8076)
"""

import os
import numpy as np

import torch
import torch.utils.data

import torchvision
import torchvision.models
import torchvision.transforms

import transforms

# from __future__ import print_function
from PIL import Image
# import os
# import os.path
# import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
# from .utils import download_url, check_integrity
from torchvision.datasets.utils import download_url, check_integrity

import os, sys
from six.moves import cPickle
import numpy as np

from keras.datasets.cifar10 import load_data as load_cifar10

from jutils import load_cifar10h_labels, load_cifar10_1


SEED = 999


class CIFAR10H(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    c10_train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    c10_test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, which_set='train',
                 transform=None, target_transform=None,
                 download=False, c10h_sample=False,
                 c10h_testsplit_percent=0.1,
                 c10h_datasplit_seed=999,
                 cv_index=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.set = which_set  # training set or test set
        self.c10h_sample = c10h_sample
        self.cv_index = cv_index

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.set in ['train','test']:
            # handles both the cifar10h (human) training and test sets
            self.c10h_data = []
            self.c10h_targets = []
            self.c10h_c10_targets = []

            downloaded_list = self.c10_test_list
            # now load the picked numpy arrays
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    self.c10h_data.append(entry['data'])
                    if 'labels' in entry:
                        self.c10h_c10_targets.extend(entry['labels'])

            self.c10h_data = np.vstack(self.c10h_data).reshape(-1, 3, 32, 32)
            self.c10h_data = self.c10h_data.transpose((0, 2, 3, 1))  # convert to HWC

            self.c10h_targets = load_cifar10h_labels(type='aggregate', 
                path='../data/cifar10h/')

            self.c10h_targets = self.c10h_targets.astype('f')
            self.c10h_targets = self.c10h_targets / self.c10h_targets.sum(axis=1)[:,None]

            # shuffle data and labels together
            c10h_rnd_idxs = np.arange(self.c10h_data.shape[0])
            # this seed is the same for self.set in ['train','test']
            np.random.seed(c10h_datasplit_seed)
            np.random.shuffle(c10h_rnd_idxs)
            self.c10h_data = self.c10h_data[c10h_rnd_idxs,:,:,:]
            self.c10h_targets = self.c10h_targets[c10h_rnd_idxs]
            self.c10h_c10_targets = np.array(self.c10h_c10_targets)
            self.c10h_c10_targets = self.c10h_c10_targets[c10h_rnd_idxs]
            self.c10h_c10_targets = list(self.c10h_c10_targets)


            # NOTE: this is from we did random splits that weren't mutually exclusive

            # split_idx = int((1 - c10h_testsplit_percent)*self.c10h_data.shape[0])

            # if self.set == 'train':
            #     self.c10h_data = self.c10h_data[:split_idx]
            #     self.c10h_targets = self.c10h_targets[:split_idx]
            #     self.c10h_c10_targets =  self.c10h_c10_targets[:split_idx]
            # elif self.set == 'test':
            #     self.c10h_data = self.c10h_data[split_idx:]
            #     self.c10h_targets = self.c10h_targets[split_idx:]
            #     self.c10h_c10_targets =  self.c10h_c10_targets[split_idx:]

            # this way we get mutually exclusive folds (k-fold crossvalidation)
            fold_size = int(c10h_testsplit_percent * self.c10h_data.shape[0])

            start = fold_size * cv_index
            end = start + fold_size

            if self.set == 'train':
                idx_keep = list(range(0, start)) + list(range(end, self.c10h_data.shape[0]))
                idx_keep = np.array(idx_keep)
                self.c10h_data = self.c10h_data[idx_keep]
                self.c10h_targets = self.c10h_targets[idx_keep]
                self.c10h_c10_targets = list(np.array(self.c10h_c10_targets)[idx_keep])
                print('Training data has shape {}'.format(self.c10h_data.shape))
            elif self.set == 'test':
                self.c10h_data = self.c10h_data[start:end]
                self.c10h_targets = self.c10h_targets[start:end]
                self.c10h_c10_targets = list(np.array(self.c10h_c10_targets)[start:end])
                print('Test data has shape {}'.format(self.c10h_data.shape))

        elif self.set == '50k':
            self._50k_data = []
            self._50k_targets = []
            downloaded_list = self.c10_train_list
            for file_name, checksum in downloaded_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    self._50k_data.append(entry['data'])
                    if 'labels' in entry:
                        self._50k_targets.extend(entry['labels'])

            self._50k_data = np.vstack(self._50k_data).reshape(-1, 3, 32, 32)
            self._50k_data = self._50k_data.transpose((0, 2, 3, 1))  # convert to HWC

        elif self.set == 'v4':
            self.v4_data, self.v4_targets = \
                load_cifar10_1(version_string='v4')
            self.v4_targets = self.v4_targets.reshape(-1).astype('int64')
        elif self.set == 'v6':
            self.v6_data, self.v6_targets = \
                load_cifar10_1(version_string='v6')
            self.v6_targets = self.v6_targets.reshape(-1).astype('int64')

        elif self.set == 'imagenet32x32':
            imagenet_path = '../data/imagenet32x32'
            ordered_class_names = [
                'airplane',
                'automobile',
                'bird',
                'cat',
                'deer',
                'dog',
                'frog',
                'horse',
                'ship',
                'truck'
            ]
            self.imagenet_data = []
            self.imagenet_targets = []

            for c, class_name in enumerate(ordered_class_names):
                print('Loading ImageNet class {} with label {}'.format(class_name, c))
                image_data = np.load(os.path.join(imagenet_path, class_name+'.npy'))
                print(' -- data shape for this class is {}'.format(image_data.shape))
                self.imagenet_targets += [c] * image_data.shape[0]
                print(' -- label count for this class is {}'.format(len([c] * image_data.shape[0])))
                self.imagenet_data.append(image_data)
            self.imagenet_data = np.concatenate(self.imagenet_data, axis=0)

            print('Final ImageNet data shape is {}'.format(self.imagenet_data.shape))
            print('Final ImageNet label count is {}'.format(len(self.imagenet_targets)))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, y_agg) where target is index of the target class.
        """
        if self.set in ['train','test']:
            img, target, c10h_c10_target = \
                self.c10h_data[index], self.c10h_targets[index], self.c10h_c10_targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            
            # if training, we have the option to sample from
            # humans their proportions as multinomial params
            if self.c10h_sample and self.set == 'train':
                # yes, you seriously have to do all this
                # to avoid numpy summing error when f32-->f64
                target = target.astype('float64')
                target /= target.sum()
                target = np.random.multinomial(1, target)
                target = target.astype('float32')

            return img, target, c10h_c10_target

        else:
            # one of the additional test sets

            if self.set == '50k':
                img, target = self._50k_data[index], self._50k_targets[index]     

            elif self.set == 'v4':
                img, target = self.v4_data[index], self.v4_targets[index]

            elif self.set == 'v6':
                img, target = self.v6_data[index], self.v6_targets[index]

            elif self.set == 'imagenet32x32':
                img, target = self.imagenet_data[index], self.imagenet_targets[index]

            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def __len__(self):
        if self.set in ['train','test']:
            return len(self.c10h_data)
        elif self.set == '50k':
            return len(self._50k_data)
        elif self.set == 'v4':
            return len(self.v4_data)
        elif self.set == 'v6':
            return len(self.v6_data)
        elif self.set == 'imagenet32x32':
            return len(self.imagenet_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.c10_train_list + self.c10_test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class Dataset(object):
    def __init__(self, config):
        self.config = config
        if config['dataset'] != 'CIFAR10H':
            self.dataset_dir = os.path.join('~/.torchvision/datasets',
                                            config['dataset'])
        else:
            self.dataset_dir = os.path.join('~/.torchvision/datasets',
                                            'CIFAR10')

        self.use_cutout = (
            'use_cutout' in config.keys()) and config['use_cutout']

        self.use_random_erasing = ('use_random_erasing' in config.keys()
                                   ) and config['use_random_erasing']

    def get_datasets(
            self, 
            c10h_sample=False, 
            c10h_testsplit_percent=0.1,
            c10h_datasplit_seed=999,
            cv_index=0
        ):

        if self.config['dataset'] != 'CIFAR10H':
            train_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                self.dataset_dir, train=True, transform=self.train_transform, download=True)
            test_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                self.dataset_dir, train=False, transform=self.test_transform, download=True)
            return train_dataset, test_dataset
        else:
            # USE CIFAR10H!
            train_dataset = CIFAR10H(
                self.dataset_dir,
                which_set='train',
                transform=self.train_transform, 
                download=True,
                c10h_sample=c10h_sample,
                c10h_datasplit_seed=c10h_datasplit_seed,
                c10h_testsplit_percent=c10h_testsplit_percent,
                cv_index=cv_index
            )
            test_dataset = CIFAR10H(
                self.dataset_dir, 
                which_set='test',
                transform=self.test_transform, 
                download=True,
                c10h_datasplit_seed=c10h_datasplit_seed,
                c10h_testsplit_percent=c10h_testsplit_percent,
                cv_index=cv_index
            )

            # cifar10 50,000 training images
            _50k_dataset = CIFAR10H(
                self.dataset_dir, 
                which_set='50k', 
                transform=self.test_transform, 
                download=True
            )

            # cifar 10.1 versions
            v4_dataset = CIFAR10H(
                self.dataset_dir,
                which_set='v4',
                transform=self.test_transform, 
            )
            v6_dataset = CIFAR10H(
                self.dataset_dir,
                which_set='v6',
                transform=self.test_transform, 
            )

            # imagenet 32x32-resized, cifar10-overlapping categories only
            imagenet32x32_dataset = CIFAR10H(
                self.dataset_dir,
                which_set='imagenet32x32',
                transform=self.test_transform, 
            )

            return train_dataset, test_dataset, _50k_dataset, v4_dataset, v6_dataset, imagenet32x32_dataset

    def _get_random_erasing_train_transform(self):
        raise NotImplementedError

    def _get_cutout_train_transform(self):
        raise NotImplementedError

    def _get_default_train_transform(self):
        raise NotImplementedError

    def _get_train_transform(self):
        if self.use_random_erasing:
            return self._get_random_erasing_train_transform()
        elif self.use_cutout:
            return self._get_cutout_train_transform()
        else:
            return self._get_default_train_transform()

class CIFARH(Dataset):
    def __init__(self, config):
        super(CIFARH, self).__init__(config)

        self.mean = np.array([0.4914, 0.4822, 0.4465])
        self.std = np.array([0.2470, 0.2435, 0.2616])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.random_erasing(
                self.config['random_erasing_prob'],
                self.config['random_erasing_area_ratio_range'],
                self.config['random_erasing_min_aspect_ratio'],
                self.config['random_erasing_max_attempt']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_cutout_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

class CIFAR(Dataset):
    def __init__(self, config):
        super(CIFAR, self).__init__(config)

        if config['dataset'] == 'CIFAR10':
            self.mean = np.array([0.4914, 0.4822, 0.4465])
            self.std = np.array([0.2470, 0.2435, 0.2616])

        elif config['dataset'] == 'CIFAR100':
            self.mean = np.array([0.5071, 0.4865, 0.4409])
            self.std = np.array([0.2673, 0.2564, 0.2762])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.random_erasing(
                self.config['random_erasing_prob'],
                self.config['random_erasing_area_ratio_range'],
                self.config['random_erasing_min_aspect_ratio'],
                self.config['random_erasing_max_attempt']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_cutout_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_test_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

def get_loader(config):

    batch_size = config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']

    cv_index = config['cv_index']

    c10h_sample = config['c10h_sample']
    c10h_testsplit_percent = config['c10h_testsplit_percent']
    c10h_datasplit_seed = config['c10h_datasplit_seed']

    dataset_name = config['dataset']
    assert dataset_name in ['CIFAR10', 'CIFAR100', 'CIFAR10H']

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset = CIFAR(config)    
    elif dataset_name == 'CIFAR10H':
        dataset = CIFARH(config)

    train_dataset, test_dataset, _50k_dataset, v4_dataset, v6_dataset, imagenet32x32_dataset = \
        dataset.get_datasets(
            c10h_sample=c10h_sample, 
            c10h_testsplit_percent=c10h_testsplit_percent,
            c10h_datasplit_seed=c10h_datasplit_seed,
            cv_index=cv_index
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    _50k_loader = torch.utils.data.DataLoader(
        _50k_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
    )
    v4_loader = torch.utils.data.DataLoader(
        v4_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
    )
    v6_loader = torch.utils.data.DataLoader(
        v6_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    imagenet32x32_loader = torch.utils.data.DataLoader(
        imagenet32x32_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return train_loader, test_loader, _50k_loader, v4_loader, v6_loader, imagenet32x32_loader
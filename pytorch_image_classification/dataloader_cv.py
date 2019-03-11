# coding: utf-8

import os, sys
import pickle
from PIL import Image
import numpy as np

import torch
import torch.utils.data
import torch.utils.data as data

import torchvision
import torchvision.models
import torchvision.transforms
from torchvision.datasets.utils import download_url, check_integrity

import transforms

# this comes from torchvision.datasets.cifar
class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
                 self,
                 root, 
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 cv_index=6, # new argument (which 10k chunk of the full 60k to use as test)
                 ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.cv_index = cv_index - 1

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        self.all_list =  self.train_list + self.test_list

        if self.train:
            downloaded_list = self.all_list[:self.cv_index] + self.all_list[self.cv_index+1:]
        else:
            downloaded_list = [self.all_list[self.cv_index]]


        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
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
        self.dataset_dir = os.path.join('~/.torchvision/datasets',
                                        config['dataset'])

        self.use_cutout = (
            'use_cutout' in config.keys()) and config['use_cutout']

        self.use_random_erasing = ('use_random_erasing' in config.keys()
                                   ) and config['use_random_erasing']

    # old version
    # def get_datasets(self):
    #     train_dataset = getattr(torchvision.datasets, self.config['dataset'])(
    #         self.dataset_dir, train=True, transform=self.train_transform, download=True)
    #     test_dataset = getattr(torchvision.datasets, self.config['dataset'])(
    #         self.dataset_dir, train=False, transform=self.test_transform, download=True)
    #     return train_dataset, test_dataset

    def get_datasets(self, cv_index=6):

        if self.config['dataset'] == 'CIFAR10':
            train_dataset = CIFAR10(
                self.dataset_dir,
                train=True,
                transform=self.train_transform,
                download=True,
                cv_index=cv_index
                )
            test_dataset = CIFAR10(
                self.dataset_dir,
                train=False,
                transform=self.test_transform,
                download=True,
                cv_index=cv_index
                )
        else:
            train_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                self.dataset_dir, train=True, transform=self.train_transform, download=True)
            test_dataset = getattr(torchvision.datasets, self.config['dataset'])(
                self.dataset_dir, train=False, transform=self.test_transform, download=True)

        return train_dataset, test_dataset

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


class MNIST(Dataset):
    def __init__(self, config):
        super(MNIST, self).__init__(config)

        self.mean = np.array([0.1307])
        self.std = np.array([0.3081])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_default_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
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
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_default_train_transform(self):
        return self._get_default_transform()

    def _get_default_test_transform(self):
        return self._get_default_transform()


class FashionMNIST(Dataset):
    def __init__(self, config):
        super(FashionMNIST, self).__init__(config)

        self.mean = np.array([0.2860])
        self.std = np.array([0.3530])

        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_default_transform()

    def _get_random_erasing_train_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(28, padding=4),
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
            torchvision.transforms.RandomCrop(28, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.normalize(self.mean, self.std),
            transforms.cutout(self.config['cutout_size'],
                              self.config['cutout_prob'],
                              self.config['cutout_inside']),
            transforms.to_tensor(),
        ])
        return transform

    def _get_default_transform(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return transform

    def _get_default_train_transform(self):
        return self._get_default_transform()

    def _get_default_test_transform(self):
        return self._get_default_transform()


def get_loader(config, cv_index=6):
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    use_gpu = config['use_gpu']

    dataset_name = config['dataset']
    assert dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']

    if dataset_name in ['CIFAR10', 'CIFAR100']:
        dataset = CIFAR(config)
    elif dataset_name == 'MNIST':
        dataset = MNIST(config)
    elif dataset_name == 'FashionMNIST':
        dataset = FashionMNIST(config)

    train_dataset, test_dataset = dataset.get_datasets(cv_index=cv_index)

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
    return train_loader, test_loader

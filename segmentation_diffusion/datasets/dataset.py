from __future__ import print_function

import errno

import torch
import torch.utils.data as data_utils
import torchvision
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
import torchvision.transforms.functional as F_transforms


import numpy as np

import os
from PIL import Image

from torchflow.data import DATA_PATH

ROOT = DATA_PATH


def fn_to_tensor(img):
    img = np.array(img)

    # Add channel to grayscale images.
    if len(img.shape) == 2:
        img = img[:, :, None]
    img = np.array(img).transpose(2, 0, 1)
    return torch.from_numpy(img)


class toTensor:
    def __init__(self):
        pass

    def __call__(self, img):
        return fn_to_tensor(img)


class BMNIST(torch.utils.data.Dataset):
    """ BINARY MNIST """
    urls = ['http://www.cs.toronto.edu/~larocheh/public/datasets/' \
            'binarized_mnist/binarized_mnist_{}.amat'.format(split)
            for split in ['train', 'valid', 'test']]
    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "train.pt"
    val_file = "val.pt"
    test_file = "test.pt"

    def __init__(self, root=ROOT, split='train', transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        if split not in ('train', 'val', 'test'):
            raise ValueError('split should be one of {train, val, test}')

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        data_file = {'train': self.training_file,
                     'val': self.val_file,
                     'test': self.test_file}[split]
        path = os.path.join(self.root, self.processed_folder, data_file)
        self.data = torch.load(path)

    def __getitem__(self, index):
        img = self.data[index]

        if self.transform is not None:
            img = img.byte()
            pil_img = F_transforms.to_pil_image(img)
            pil_img = self.transform(pil_img)
            img = fn_to_tensor(pil_img)
            img = img.long()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        processed_folder = os.path.join(self.root, self.processed_folder)
        train_path = os.path.join(processed_folder, self.training_file)
        val_path = os.path.join(processed_folder, self.val_file)
        test_path = os.path.join(processed_folder, self.test_file)
        return os.path.exists(train_path) and os.path.exists(val_path) and \
            os.path.exists(test_path)

    def _read_raw_image_file(self, path):

        raw_file = os.path.join(self.root, self.raw_folder, path)
        all_images = []
        with open(raw_file) as f:
            for line in f:
                im = [int(x) for x in line.strip().split()]
                assert len(im) == 28 ** 2
                all_images.append(im)
        return torch.from_numpy(np.array(all_images)).view(-1, 28, 28)

    def download(self):
        """
        Download the BMNIST data if it doesn't exist in
        processed_folder already.
        """
        if self._check_exists():
            return

        # Create folders
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, root=os.path.join(self.root, self.raw_folder),
                         filename=filename, md5=None)

        # process and save as torch files
        print('Processing raw data..')

        training_set = self._read_raw_image_file('binarized_mnist_train.amat')
        val_set = self._read_raw_image_file('binarized_mnist_valid.amat')
        test_set = self._read_raw_image_file('binarized_mnist_test.amat')

        processed_dir = os.path.join(self.root, self.processed_folder)
        with open(os.path.join(processed_dir, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(processed_dir, self.val_file), 'wb') as f:
            torch.save(val_set, f)
        with open(os.path.join(processed_dir, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Completed data download.')


def load_bmnist(batch_size, download=True, **kwargs):
    train_transforms = transforms.Compose([
        transforms.Pad(1, padding_mode='edge'),
        transforms.RandomCrop(28),
        toTensor()
    ])

    test_transforms = transforms.Compose([
        toTensor()
    ])

    root = ROOT + '/bmnist'
    train_set = BMNIST(root, 'train', train_transforms, download)
    val_set = BMNIST(root, 'val', test_transforms, download)
    test_set = BMNIST(root, 'test', test_transforms, download)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=10)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=10)

    return trainloader, valloader, testloader


def load_svhn(args, **kwargs):
    # set args
    args.input_size = [3, 32, 32]
    args.num_classes = 256

    train_transform = transforms.Compose([
        transforms.Pad(1, padding_mode='edge'),
        transforms.RandomCrop(32),
        toTensor()
    ])

    test_transform = transforms.Compose([
        toTensor()
    ])

    root = ROOT + '/svhn'
    train_data = torchvision.datasets.SVHN(
        root, split='train', transform=train_transform, target_transform=None,
        download=True)

    stepsize = len(train_data) // 10000
    val_indcs = np.arange(0, len(train_data), stepsize)
    train_idcs = np.setdiff1d(np.arange(len(train_data)), val_indcs)

    print('SVHN division: train: {}, val: {}, every {}th index is val.'.format(
        len(train_idcs), len(val_indcs), stepsize
    ))

    train_data = torch.utils.data.Subset(
        train_data, indices=train_idcs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_data = torchvision.datasets.SVHN(
        root, split='train', transform=test_transform, target_transform=None,
        download=True)

    val_data = torch.utils.data.Subset(
        val_data, indices=val_indcs)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)

    test_data = torchvision.datasets.SVHN(
        root, split='test', transform=test_transform, target_transform=None,
        download=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, args
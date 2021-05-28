#!/usr/bin/env python

'''
Example of the blocksparse transformer on enwik8.

To download data:

wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip -d /tmp
'''

import argparse
import numpy       as np
import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
# from mpi4py import MPI
from .vocab import Vocab


DATA_PATH = './datasets'


class EnWik8Dataset(Dataset):
    """


    """

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'enwik8')
        self.seq_len = seq_len
        self.split = split

        if not os.path.exists(self.raw_file):
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')

        # Get vocabulary
        self.vocab = Vocab()
        vocab_file = os.path.join(self.root, 'vocab.json')
        # if os.path.exists(vocab_file):
        #     self.vocab.load_json(self.root)
        # else:
        stoi = self._create_stoi()
        self.vocab.fill(stoi)
            # self.vocab.save_json(self.root)

        # Preprocess data
        # if not os.path.exists(self.processed_file(split)):
        #     self._preprocess_data(split)

        # Load data
        self.data = self._preprocess_data(split)

        # self.data = torch.load(self.processed_file(split))

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def _create_stoi(self):
        # Just a simple identity conversion for 8bit (byte)-valued chunks.
        stoi = {i: i for i in range(256)}
        return stoi

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = zipfile.ZipFile(self.raw_file).read('enwik8')

        n_train = int(90e6)
        n_valid = int(5e6)
        n_test = int(5e6)

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:n_train]
        elif split == 'valid':
            rawdata = rawdata[n_train:n_train+n_valid]
        elif split == 'test':
            rawdata = rawdata[n_train+n_valid:n_train+n_valid+n_test]

        # Encode characters
        data = torch.tensor([self.vocab.stoi[s] for s in rawdata])

        # Split into chunks %TODO create version with changing offset.
        # data = data[:self.seq_len*(len(data)//self.seq_len)]
        data = data.reshape(-1, self.seq_len)

        return data
        # print(rawdata)
        # Save processed data
        # torch.save(data, self.processed_file(split))

    @property
    def raw_file(self):
        return os.path.join(self.root, 'enwik8.zip')

    # def processed_file(self, split):
    #     return os.path.join(self.root, 'processed_{}.pt'.format(split))

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading enwik8...')
        url = 'http://mattmahoney.net/dc/enwik8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.raw_file)
        print('Saved to {}'.format(self.raw_file))

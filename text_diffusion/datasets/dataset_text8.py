import torch
import os
import json
import zipfile
import urllib.request
from torch.utils.data import Dataset
from .vocab import Vocab


DATA_PATH = './datasets'


class Text8Dataset(Dataset):
    """
    The text8 dataset consisting of 100M characters (with vocab size 27).
    We here split the dataset into (90M, 5M, 5M) characters for
    (train, val, test) as in [1,2,3].

    The sets are then split into chunks of equal length as specified by `seq_len`.
    The default is 256, corresponding to what was used in [1]. Other choices
    include 180, as [2] reports using.

    [1] Discrete Flows: Invertible Generative Models of Discrete Data
        Tran et al., 2019, https://arxiv.org/abs/1905.10347
    [2] Architectural Complexity Measures of Recurrent Neural Networks
        Zhang et al., 2016, https://arxiv.org/abs/1602.08210
    [3] Subword Language Modeling with Neural Networks
        Mikolov et al., 2013, http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
    """

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(root, 'text8')
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
        if os.path.exists(vocab_file):
            self.vocab.load_json(self.root)
        else:
            stoi = self._create_stoi()
            self.vocab.fill(stoi)
            self.vocab.save_json(self.root)

        # Preprocess data
        if not os.path.exists(self.processed_file(split)):
            self._preprocess_data(split)

        # Load data
        self.data = torch.load(self.processed_file(split))

    def __getitem__(self, index):
        return self.data[index], self.seq_len

    def __len__(self):
        return len(self.data)

    def _create_stoi(self):
        rawdata = zipfile.ZipFile(self.raw_file).read('text8').decode('utf-8')
        s = sorted(list(set(rawdata)))
        stoi = {s[i]: i for i in range(len(s))}
        return stoi

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = zipfile.ZipFile(self.raw_file).read('text8').decode('utf-8')

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:90000000]
        elif split == 'valid':
            rawdata = rawdata[90000000:95000000]
        elif split == 'test':
            rawdata = rawdata[95000000:]

        # Encode characters
        data = torch.tensor([self.vocab.stoi[s] for s in rawdata])

        # Split into chunks
        data = data[:self.seq_len*(len(data)//self.seq_len)]
        data = data.reshape(-1, self.seq_len)

        # Save processed data
        torch.save(data, self.processed_file(split))

    @property
    def raw_file(self):
        return os.path.join(self.root, 'text8.zip')

    def processed_file(self, split):
        return os.path.join(self.root, 'processed_{}.pt'.format(split))

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading text8...')
        url = 'http://mattmahoney.net/dc/text8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.raw_file)
        print('Saved to {}'.format(self.raw_file))

import os
import errno
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
from PIL import Image

from .cityscapes import map_id_to_category_id
from .cityscapes import ToTensorNoNorm, onehot_segmentation_to_img
import os

from .cityscapes import cityscapes_indices_segmentation_to_img, \
    cityscapes_only_categories_indices_segmentation_to_img


ROOT = os.path.dirname(os.path.abspath(__file__))


class CityscapesFast(data.Dataset):
    def __init__(self, root=ROOT, split='train', resolution=(32, 64), transform=None, only_categories=False):
        assert resolution in [(32, 64), (128, 256)]

        H, W = resolution

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.only_categories = only_categories

        if split not in ('train', 'val', 'test'):
            raise ValueError('split should be one of {train, val, test}')

        if not self._check_exists(H, W):
            raise RuntimeError('Dataset not found (or incomplete) at {}'.format(self.root))

        self.data = torch.from_numpy(
            np.load(join(self.root, 'preprocessed', split + f'_{H}x{W}.npy')))

    def __getitem__(self, index):
        img = self.data[index]

        img = img.long()

        if self.only_categories:
            img = map_id_to_category_id[img]

        if self.transform:
            assert img.size(0) == 1
            img = img[0]

            img = Image.fromarray(img.numpy().astype('uint8'))
            img = self.transform(img)

            img = np.array(img)

            img = torch.tensor(img).long()

            img = img.unsqueeze(0)

        # mock_label = torch.zeros_like(img[0, 0]).int()

        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self, H, W):
        train_path = os.path.join(self.root, 'preprocessed', f'train_{H}x{W}.npy')
        val_path = os.path.join(self.root, 'preprocessed', f'val_{H}x{W}.npy')
        test_path = os.path.join(self.root, 'preprocessed', f'test_{H}x{W}.npy')

        return os.path.exists(train_path) and os.path.exists(val_path) and \
            os.path.exists(test_path)


def get_categories(args, root=ROOT):
    return get(args, root=root, only_categories=True)


def get(args, root=ROOT, only_categories=False):
    train_set = CityscapesFast(
        root=root, split='train', transform=None,
        only_categories=only_categories)
    val_set = CityscapesFast(
        root=root, split='val', transform=None,
        only_categories=only_categories)
    test_set = CityscapesFast(
        root=root, split='test', transform=None,
        only_categories=only_categories)

    # train_set = torch.utils.data.Subset(data_set, torch.arange(0, 2500))
    # val_set = torch.utils.data.Subset(data_set, torch.arange(2500, 2975))


    # test_set = Cityscapes(
    #     root=root, split='test', mode='fine', target_type='instance',
    #     transform=data_transforms, target_transform=None, transforms=None)

    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=10,
                                              drop_last=True)
    valloader = torch.utils.data.DataLoader(val_set,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=10)
    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=10)

    # for batch, _ in trainloader:
    #     img = onehot_segmentation_to_img(batch.long(), colors=COLORS)
    #     torchvision.utils.save_image(
    #         img / 256., 'cityscapes.png', nrow=10, padding=2,
    #         normalize=False, range=None, scale_each=False, pad_value=0,
    #         format=None)
    #     break

    args.data_size = (1, H, W)
    args.variable_type = 'categorical'
    args.data_channels = 1

    if only_categories:
        args.num_classes = 8
    else:
        args.num_classes = 34

    return trainloader, valloader, testloader, args


if __name__ == '__main__':
    class Args:
        batch_size = 50

    args = Args()
    trainloader, valloader, testloader, args = get_categories(args, './')

    for batch, _ in trainloader:
        img = cityscapes_only_categories_indices_segmentation_to_img(batch.long())
        torchvision.utils.save_image(
            img / 256., 'cityscapes.png', nrow=10, padding=2,
            normalize=False, range=None, scale_each=False, pad_value=0,
            format=None)
        break

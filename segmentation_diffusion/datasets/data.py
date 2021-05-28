import math
import torch
import numpy as np
from torch.utils.data import DataLoader
# from torchflow.data.loaders.nde.image import MNIST
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, \
    CenterCrop, RandomCrop, Compose, ToPILImage, ToTensor

from cityscapes.cityscapes_fast import CityscapesFast, \
    cityscapes_indices_segmentation_to_img, \
    cityscapes_only_categories_indices_segmentation_to_img

dataset_choices = {
    'cityscapes_coarse', 'cityscapes_fine',
    'cityscapes_coarse_large', 'cityscapes_fine_large'}


def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='cityscapes_coarse',
                        choices=dataset_choices)

    # Train params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)


def get_plot_transform(args):
    if args.dataset in ('cityscapes_coarse', 'cityscapes_coarse_large'):
        return cityscapes_only_categories_indices_segmentation_to_img

    elif args.dataset in ('cityscapes_fine', 'cityscapes_fine_large'):
        return cityscapes_indices_segmentation_to_img

    else:
        def identity(x):
            return x
        return identity


def get_data_id(args):
    return '{}'.format(args.dataset)


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    # data_shape = get_data_shape(args.dataset)

    if args.dataset == 'cityscapes_coarse':
        data_shape = (1, 32, 64)
        num_classes = 8
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (32, 64))
        pil_transforms = Compose(pil_transforms)
        train = CityscapesFast(split='train', resolution=(32, 64), transform=pil_transforms, only_categories=True)
        test = CityscapesFast(split='test', resolution=(32, 64), transform=pil_transforms, only_categories=True)
    elif args.dataset == 'cityscapes_fine':
        data_shape = (1, 32, 64)
        num_classes = 34
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (32, 64))
        pil_transforms = Compose(pil_transforms)
        train = CityscapesFast(split='train', resolution=(32, 64), transform=pil_transforms, only_categories=False)
        test = CityscapesFast(split='test', resolution=(32, 64), transform=pil_transforms, only_categories=False)

    elif args.dataset == 'cityscapes_coarse_large':
        data_shape = (1, 128, 256)
        num_classes = 8
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (128, 256))
        pil_transforms = Compose(pil_transforms)

        train = CityscapesFast(
            split='train', resolution=(128, 256), transform=pil_transforms,
            only_categories=True)
        test = CityscapesFast(
            split='test', resolution=(128, 256), transform=pil_transforms,
            only_categories=True)

    elif args.dataset == 'cityscapes_fine_large':
        data_shape = (1, 128, 256)
        num_classes = 34
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (128, 256))
        pil_transforms = Compose(pil_transforms)
        train = CityscapesFast(
            split='train', resolution=(128, 256), transform=pil_transforms,
            only_categories=False)
        test = CityscapesFast(
            split='test', resolution=(128, 256), transform=pil_transforms,
            only_categories=False)

    else:
        raise ValueError

    # Data Loader
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, eval_loader, data_shape, num_classes


def get_augmentation(augmentation, dataset, data_shape):
    h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'shift':
        pad_h, pad_w = int(0.07 * h), int(0.07 * w)
        if 'cityscapes' in dataset and 'large' in dataset:
            # Annoying, cityscapes images have a 3-border around every image.
            # This messes up shift augmentation and needs to be dealt with.
            assert h == 128 and w == 256
            print('Special cityscapes transform')
            pad_h, pad_w = int(0.075 * h), int(0.075 * w)
            pil_transforms = [CenterCrop((h - 2, w - 2)),
                              RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h - 2, w - 2)),
                              Pad((1, 1), padding_mode='constant', fill=3)]

        else:
            pil_transforms = [RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h, w))]
    elif augmentation == 'neta':
        assert h == w
        pil_transforms = [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    elif augmentation == 'eta':
        assert h == w
        pil_transforms = [RandomHorizontalFlip(),
                          Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]

    # torchvision.transforms.s
    return pil_transforms


def get_data_shape(dataset):
    if dataset == 'bmnist':
        return (28, 28)

    elif dataset == 'mnist_1bit':
        return (28, 28)

import os
import errno

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
from PIL import Image
# from data.utils import ToTensorNoNorm, onehot_segmentation_to_img, \
#     indices_segmentation_to_img
import imageio


import json
import os
from collections import namedtuple
import zipfile

from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image

ROOT = None

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

map_id_to_category_id = [x.category_id for x in classes]
map_id_to_category_id = torch.tensor(map_id_to_category_id)


# map_id_to_color = [(x.id, x.color) for x in classes]
COLORS = [x.color for x in classes]
COLORS = torch.tensor(COLORS)
# fn_map_np = np.vectorize(fn_map)

coarse_COLORS = torch.tensor(
    [(0, 0, 0),
     (128, 64, 128),
     (70, 70, 70),
     (153, 153, 153),
     (107, 142, 35),
     (70, 130, 180),
     (220, 20, 60),
     (0, 0, 142)
     ]
)

def onehot_segmentation_to_img(onehot, colors):
    indices = torch.argmax(onehot, dim=1)
    return indices_segmentation_to_img(indices, colors)


def indices_segmentation_to_img(indices, colors):
    if indices.size(1) == 1:
        # Remove single channel axis.
        indices = indices[:, 0]
    rgbs = colors[indices]
    rgbs = rgbs.permute(0, 3, 1, 2)
    return rgbs


class ToTensorNoNorm():
    def __call__(self, X_i):
        X_i = np.array(X_i)

        if len(X_i.shape) == 2:
            # Add channel dim.
            X_i = X_i[:, :, None]

        return torch.from_numpy(np.array(X_i, copy=False)).permute(2, 0, 1)


class Cityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts

    def __init__(self, root=ROOT, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None,
                 only_categories=False):
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.only_categories = only_categories

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.targets_dir):
            raise ValueError(f'no path {self.targets_dir}')
        # if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
        #
        #     if split == 'train_extra':
        #         image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
        #     else:
        #         image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))
        #
        #     if self.mode == 'gtFine':
        #         target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
        #     elif self.mode == 'gtCoarse':
        #         target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))
        #
        #     if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
        #         extract_archive(from_path=image_dir_zip, to_path=self.root)
        #         extract_archive(from_path=target_dir_zip, to_path=self.root)
        #     else:
        #         raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
        #                            ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.targets_dir):
            # img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(target_dir):
                # print(file_name)
                if file_name[-5:] == '.json':
                    target_types = []

                    if mode == 'fine':
                        prefix = file_name.split('_gtFine_polygons.json')[0]
                    elif mode == 'coarse':
                        prefix = file_name.split('_gtCoarse_polygons.json')[0]
                    else:
                        raise ValueError

                    for t in self.target_type:
                        target_name = '{}_{}'.format(
                            prefix, self._get_target_suffix(self.mode, t))
                        target_types.append(
                            os.path.join(target_dir, target_name))

                # self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(target_types)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        # image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                # target = imageio.imread(self.targets[index][i])
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]
        # with np.printoptions(threshold=np.inf):
        #     print(np.array(target).max())
        #     print(np.array(target))

        if self.transform is not None:
            target = self.transform(target).long()

        if self.only_categories:
            target = map_id_to_category_id[target]

        dummy = torch.zeros_like(target[0, 0])

        return target, dummy

    def __len__(self):
        return len(self.targets)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


def cityscapes_indices_segmentation_to_img(img):
    return indices_segmentation_to_img(img, colors=COLORS)


def cityscapes_only_categories_indices_segmentation_to_img(img):
    return indices_segmentation_to_img(img, colors=coarse_COLORS)


def get_categories(args, root='/local-ssd/emiel/cityscapes'):
    return get(args, root=root, only_categories=True)


def get(args, root='/local-ssd/emiel/cityscapes', only_categories=False):
    H, W = 32, 64

    data_transforms = transforms.Compose([
        torchvision.transforms.Resize((H, W), interpolation=0),
        ToTensorNoNorm()])
    data_set = Cityscapes(
        root=root, split='train', mode='fine', target_type='semantic',
        transform=data_transforms, target_transform=None, transforms=None,
        only_categories=only_categories)
    test_set = Cityscapes(
        root=root, split='val', mode='fine', target_type='semantic',
        transform=data_transforms, target_transform=None, transforms=None,
        only_categories=only_categories)

    train_set = torch.utils.data.Subset(data_set, torch.arange(0, 2500))
    val_set = torch.utils.data.Subset(data_set, torch.arange(2500, 2975))


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
        batch_size = 100

    args = Args()
    trainloader, valloader, testloader, args = get(args, '../data/cityscapes')

import os

import torch
import numpy as np
import torchvision
from torchvision import transforms

from cityscapes import Cityscapes
from cityscapes import ToTensorNoNorm


def main(H, W):
    data_transforms = transforms.Compose([
        torchvision.transforms.Resize((H, W), interpolation=0),
        ToTensorNoNorm()])

    data_set = Cityscapes(
            root='./', split='train', mode='fine', target_type='semantic',
            transform=data_transforms, target_transform=None, transforms=None,
            only_categories=False)
    test_set = Cityscapes(
        root='./', split='val', mode='fine', target_type='semantic',
        transform=data_transforms, target_transform=None, transforms=None,
        only_categories=False)

    train_set = torch.utils.data.Subset(data_set, torch.arange(0, 2500))
    val_set = torch.utils.data.Subset(data_set, torch.arange(2500, 2975))

    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)

    train_tensor, val_tensor, test_tensor = \
        torch.zeros((n_train, 1, H, W)).long(), \
        torch.zeros((n_val, 1, H, W)).long(), \
        torch.zeros((n_test, 1, H, W)).long()

    datasets = [train_set, val_set, test_set]
    tensors = [train_tensor, val_tensor, test_tensor]
    names = [f'train_{H}x{W}', f'val_{H}x{W}', f'test_{H}x{W}']
    for name, dataset, target_tensor in zip(names, datasets, tensors):
        print(f'Loading {name}')
        min_val, max_val = 0, 0
        for i, item in enumerate(dataset):
            x, y = item
            target_tensor[i] = x

            if min_val > x.min():
                min_val = x.min()
            if max_val < x.max():
                max_val = x.max()

        print(f'min {min_val} max {max_val}')

    root = './preprocessed'
    os.makedirs(root, exist_ok=True)

    train = train_tensor.numpy().astype('uint8')
    np.save(os.path.join(root, f'train_{H}x{W}'), train, allow_pickle=True, fix_imports=True)

    val = val_tensor.numpy().astype('uint8')
    np.save(os.path.join(root, f'val_{H}x{W}'), val, allow_pickle=True, fix_imports=True)

    test = test_tensor.numpy().astype('uint8')
    np.save(os.path.join(root, f'test_{H}x{W}'), test, allow_pickle=True, fix_imports=True)

    # Test if works.
    train = np.load(os.path.join(root, f'train_{H}x{W}.npy'))
    print(train.shape)

    val = np.load(os.path.join(root, f'val_{H}x{W}.npy'))
    print(val.shape)

    test = np.load(os.path.join(root, f'test_{H}x{W}.npy'))
    print(test.shape)


if __name__ == '__main__':
    main(32, 64)
    main(128, 256)

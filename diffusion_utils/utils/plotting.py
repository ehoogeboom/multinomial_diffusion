import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def get_image_grid(images, nrow=8, padding=2):
    '''
    Get a plotting-friendly grid image from images.

    Args:
        images: Tensor, shape (b, c, h, w)
        nrow: int, the number of images per row.
        padding: int, the number of padding pixels.

    Returns:
        image_grid: numpy array, shape (H,W,c), where H and W are the size of the grid image.
    '''
    image_grid = vutils.make_grid(images, nrow=nrow, padding=padding)
    image_grid = image_grid.permute([1,2,0]).detach().cpu().numpy()
    return image_grid



def plot_quantized_images(images, num_bits=8, nrow=8, padding=2):
    '''
    Plot quantized images.

    Args:
        images: Tensor, shape (b, c, h, w)
        num_bits: int, the number of bits for the quantized image.
        nrow: int, the number of images per row.
        padding: int, the number of padding pixels.
    '''
    image_grid = get_image_grid(images.float()/(2**num_bits - 1), nrow=nrow, padding=padding)
    plt.figure()
    plt.imshow(image_grid)
    plt.show()

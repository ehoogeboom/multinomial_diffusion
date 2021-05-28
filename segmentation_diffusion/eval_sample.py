import os
import math
import torch
import numpy as np
import pickle
import argparse
import torchvision.utils as vutils
from diffusion_utils.utils import add_parent_path
import torchvision
import imageio

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args, get_plot_transform

# Model
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--nrow', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

train_loader, _, data_shape, num_classes = get_data(args)
minibatch_data = None
for minibatch_data in train_loader:
    break
assert minibatch_data is not None

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)

if torch.cuda.is_available():
    checkpoint = torch.load(path_check)
else:
    checkpoint = torch.load(path_check, map_location='cpu')
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.cuda()

print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))


############
## Sample ##
############
plot_transform = get_plot_transform(args)


def batch_samples_to_grid(batch):
    if len(batch.size()) == 3:
        batch = batch.unsqueeze(1)

    batch = plot_transform(batch).to(torch.uint8)

    grid = torchvision.utils.make_grid(
        batch, nrow=5, padding=2, normalize=False)

    grid = grid.permute(1, 2, 0)
    return grid


path_samples = '{}/samples/sample_ep{}_s{}.png'.format(eval_args.model, checkpoint['current_epoch'], eval_args.seed)
if not os.path.exists(os.path.dirname(path_samples)):
    os.mkdir(os.path.dirname(path_samples))

path_data_samples = '{}/samples/data.png'.format(eval_args.model,
                                                 checkpoint[
                                                     'current_epoch'],
                                                 eval_args.seed)


# imageio.imsave(path_samples, batch_samples_to_grid(minibatch_data))

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = model.to(device)
# model = model.eval()
# if eval_args.double: model = model.double()
#
# samples = model.sample(eval_args.samples).cpu().float()/(2**args.num_bits - 1)
# vutils.save_image(samples, fp=path_samples, nrow=eval_args.nrow)

chain_samples = eval_args.samples
with torch.no_grad():
    samples_chain = model.sample_chain(chain_samples)


images = []
for samples_i in samples_chain:
    grid = batch_samples_to_grid(samples_i)
    images.append(grid)

images = list(reversed(images))


def chain_linspace(chain, num_steps=150, repeat_last=10):
    out = []
    for i in np.linspace(0, len(chain)-1, num_steps):
        idx = int(i)
        if idx >= len(chain):
            print('index too big')
            idx = idx - 1
        out.append(chain[idx])

    # So that the animation stalls at the final output.
    for i in range(repeat_last):
        out.append(chain[-1])
    return out


images = chain_linspace(images)

# images.extend([images[-1], images[-1], images[-1], images[-1], images[-1]])

# images = np.array(images)
# images = images[np.arange(0, len(images), 10)]


imageio.mimsave(path_samples[:-4] + '_chain.gif', images)
imageio.imsave(path_samples, images[-1])

# from pygifsicle import optimize
# optimize(path_samples[:-4] + "_chain.gif")

import os
import math
import time

import imageio
import numpy as np
import torch
import pickle
import argparse
from diffusion_utils.utils import add_parent_path
from diffusion_utils.diffusion_multinomial import \
    index_to_log_onehot, log_onehot_to_index

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Model
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--length', type=int, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--benchmark', type=eval, default=False)
eval_args = parser.parse_args()
assert eval_args.length is not None, 'Currently, length has to be specified.'

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

train_loader, eval_loader, data_shape, num_classes = get_data(args)

# for batch in eval_loader:
#
#     text, length = batch
#
#     text = \
#         train_loader.dataset.vocab.decode(text, length)
#     print(text)
#     quit()

ground_truth = 'mexico city the aztec stadium estadio azteca home of club' \
               ' america is one of the world s largest stadiums with capacity' \
               ' to seat approximately one one zero zero zero zero fans mexico' \
               ' hosted the football world cup in one nine seven zero and one' \
               ' nine eight six'
print(len(ground_truth), 'len ground truth')
ground_truth = [ground_truth]

ground_truth_encoded, ground_truth_length = train_loader.dataset.vocab.encode(ground_truth)

corrupted = \
    'mexico citi the aztec stadium estadio azteca home of clup' \
    ' amerika is one of the world s largest stadioms with capakity' \
    ' to seat approsimately one one zeto zero zero zero fans mexico' \
    ' hosted the footpall wolld cup in one nine zeven zero and one' \
    ' nyne eiggt six'

corrupted = [corrupted]

corrupted_encoded, corrupted_length = \
    train_loader.dataset.vocab.encode(corrupted)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape, num_classes=num_classes)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

############
## Sample ##
############

path_samples = os.path.join(eval_args.model, 'samples/ground_truth_ep{}.txt'.format(checkpoint['current_epoch']))
if not os.path.exists(os.path.dirname(path_samples)):
    os.mkdir(os.path.dirname(path_samples))

with open(path_samples, 'w') as f:
    f.write('\n\n\n'.join(ground_truth))

path_samples = os.path.join(eval_args.model, 'samples/corrupted_ep{}.txt'.format(checkpoint['current_epoch']))
with open(path_samples, 'w') as f:
    f.write('\n\n\n'.join(corrupted))


time = torch.ones(1).long() * 0
log_x = index_to_log_onehot(corrupted_encoded, num_classes=27)
log_pred = model.base_dist.predict_start(log_x, t=time)
corrected_encoded = log_onehot_to_index(log_pred)

corrected = \
    train_loader.dataset.vocab.decode(corrected_encoded, corrupted_length)

path_samples = os.path.join(eval_args.model, 'samples/corrected_ep{}.txt'.format(checkpoint['current_epoch']))
with open(path_samples, 'w') as f:
    f.write('\n\n\n'.join(corrected))

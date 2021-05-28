import torch
import argparse

# Plot
import matplotlib.pyplot as plt

# Data
from data import get_data, add_data_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
add_data_args(parser)
parser.add_argument('--print_num', type=int, default=4)
args = parser.parse_args()

torch.manual_seed(0)

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, num_classes = get_data(args)

##############
## Sampling ##
##############

print('Train Batches:', len(train_loader))
print('Train Batch[0]:')
tensor, length = next(iter(train_loader))
print(tensor.shape, tensor.min(), tensor.max())
print(length)
text = train_loader.dataset.vocab.decode(tensor, length)
for i, s in enumerate(text):
    print('\nSample {}:'.format(i))
    print(length[i])
    print(tensor[i])
    print(s)
    if args.print_num==i+1: break

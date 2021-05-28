import os
import math
import torch
import numpy as np
import pickle
import argparse
import torchvision.utils as vutils
from diffusion_utils.utils import add_parent_path
from diffusion_utils.loss import dataset_elbo_bpd, dataset_iwbo_bpd
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
parser.add_argument('--kbs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--repetitions', type=int, default=100)
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

train_loader, eval_loader, data_shape, num_classes = get_data(args)

# Adjust args
args.batch_size = eval_args.batch_size

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)
checkpoint = torch.load(path_check)
model.load_state_dict(checkpoint['model'])
print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

############
## Loglik ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()
if eval_args.double: model = model.double()

eval_str = 'elbo'
bpds = []
for i in range(eval_args.repetitions):
    bpd = dataset_elbo_bpd(model, eval_loader, device=device, double=eval_args.double)
    bpds.append(bpd)

bpd = np.mean(bpds)


path_loglik = '{}/loglik/{}_ep{}.txt'.format(eval_args.model, eval_str, checkpoint['current_epoch'])
if not os.path.exists(os.path.dirname(path_loglik)):
    os.mkdir(os.path.dirname(path_loglik))

print(f'Eval bpd {bpd}')

with open(path_loglik, 'w') as f:
    f.write(str(bpd))

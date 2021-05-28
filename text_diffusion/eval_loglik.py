import os
import math
import torch
import pickle
import argparse
from diffusion_utils.utils import add_parent_path

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
parser.add_argument('--model', type=str, default=None, required=True)
parser.add_argument('--validation', type=eval, default=False)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--double', type=eval, default=False)
parser.add_argument('--seed', type=int, default=0)
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

# Adjust args
args.batch_size = eval_args.batch_size
args.validation = eval_args.validation

train_loader, eval_loader, data_shape, num_classes = get_data(args)

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
## Loglik ##
############

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model = model.eval()
if eval_args.double: model = model.double()

dset_str = 'valid' if eval_args.validation else 'test'
eval_str = 'loglik'
# Compute loglik
with torch.no_grad():
    bpd = 0.0
    count = 0
    for i, (x, length) in enumerate(eval_loader):
        if eval_args.double: x = x.double()
        x = x.to(device)
        num_elem = length.sum()
        bpd_batch = - model.log_prob(x).sum() / (math.log(2) * num_elem)
        bpd += bpd_batch.cpu().item() * len(x)
        count += len(x)
        print('{}/{}'.format(i+1, len(eval_loader)), bpd/count, end='\r')
bpd = bpd / count

path_loglik = '{}/loglik/{}_{}_ep{}.txt'.format(eval_args.model, dset_str, eval_str, checkpoint['current_epoch'])
if not os.path.exists(os.path.dirname(path_loglik)):
    os.mkdir(os.path.dirname(path_loglik))

with open(path_loglik, 'w') as f:
    f.write(str(bpd))

print(f'Final test bpd {bpd}')

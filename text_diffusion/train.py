import torch
import argparse
from diffusion_utils.utils import add_parent_path, set_seeds

# Exp
from experiment import Experiment, add_exp_args

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args

# Model
from model import get_model, get_model_id, add_model_args

# Optim
from diffusion_utils.expdecay import get_optim, get_optim_id, add_optim_args

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0)
add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
set_seeds(args.seed)

##################
## Specify data ##
##################

train_loader, eval_loader, data_shape, num_classes = get_data(args)
data_id = get_data_id(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape, num_classes=num_classes)
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############

exp = Experiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

exp.run()

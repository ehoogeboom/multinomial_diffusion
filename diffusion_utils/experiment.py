import torch
from diffusion_utils.utils import get_args_table, clean_dict

# Path
import os
import time
import pathlib
HOME = str(pathlib.Path.home())

# Experiment
from diffusion_utils import BaseExperiment
from diffusion_utils.base import DataParallelDistribution

#  Logging frameworks
from torch.utils.tensorboard import SummaryWriter
import wandb


def add_exp_args(parser):

    # Train params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
    parser.add_argument('--resume', type=str, default=None)

    # Logging params
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--check_every', type=int, default=None)
    parser.add_argument('--log_tb', type=eval, default=True)
    parser.add_argument('--log_wandb', type=eval, default=True)
    parser.add_argument('--log_home', type=str, default=None)


class DiffusionExperiment(BaseExperiment):
    no_log_keys = ['project', 'name',
                   'log_tb', 'log_wandb',
                   'check_every', 'eval_every',
                   'device', 'parallel'
                   'pin_memory', 'num_workers']

    def __init__(self, args,
                 data_id, model_id, optim_id,
                 train_loader, eval_loader,
                 model, optimizer, scheduler_iter, scheduler_epoch):
        if args.log_home is None:
            self.log_base = os.path.join(HOME, 'log', 'flow')
        else:
            self.log_base = os.path.join(args.log_home, 'log', 'flow')

        # Edit args
        if args.eval_every is None:
            args.eval_every = args.epochs
        if args.check_every is None:
            args.check_every = args.epochs
        if args.name is None:
            args.name = time.strftime("%Y-%m-%d_%H-%M-%S")
        if args.project is None:
            args.project = '_'.join([data_id, model_id])

        # Move model
        model = model.to(args.device)
        if args.parallel == 'dp':
            model = DataParallelDistribution(model)

        # Init parent
        super(DiffusionExperiment, self).__init__(model=model,
                                                  optimizer=optimizer,
                                                  scheduler_iter=scheduler_iter,
                                                  scheduler_epoch=scheduler_epoch,
                                                  log_path=os.path.join(self.log_base, data_id, model_id, optim_id, args.name),
                                                  eval_every=args.eval_every,
                                                  check_every=args.check_every)

        # Store args
        self.create_folders()
        self.save_args(args)
        self.args = args

        # Store IDs
        self.data_id = data_id
        self.model_id = model_id
        self.optim_id = optim_id

        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Init logging
        args_dict = clean_dict(vars(args), keys=self.no_log_keys)
        if args.log_tb:
            self.writer = SummaryWriter(os.path.join(self.log_path, 'tb'))
            self.writer.add_text("args", get_args_table(args_dict).get_html_string(), global_step=0)
        if args.log_wandb:
            wandb.init(config=args_dict, project=args.project, id=args.name, dir=self.log_path)

    def log_fn(self, epoch, train_dict, eval_dict):

        # Tensorboard
        if self.args.log_tb:
            for metric_name, metric_value in train_dict.items():
                self.writer.add_scalar('base/{}'.format(metric_name), metric_value, global_step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    self.writer.add_scalar('eval/{}'.format(metric_name), metric_value, global_step=epoch+1)

        # Weights & Biases
        if self.args.log_wandb:
            for metric_name, metric_value in train_dict.items():
                wandb.log({'base/{}'.format(metric_name): metric_value}, step=epoch+1)
            if eval_dict:
                for metric_name, metric_value in eval_dict.items():
                    wandb.log({'eval/{}'.format(metric_name): metric_value}, step=epoch+1)

    def resume(self):
        resume_path = os.path.join(self.log_base, self.data_id, self.model_id, self.optim_id, self.args.resume, 'check')
        self.checkpoint_load(resume_path)
        for epoch in range(self.current_epoch):
            train_dict = {}
            for metric_name, metric_values in self.train_metrics.items():
                train_dict[metric_name] = metric_values[epoch]
            if epoch in self.eval_epochs:
                eval_dict = {}
                for metric_name, metric_values in self.eval_metrics.items():
                    eval_dict[metric_name] = metric_values[self.eval_epochs.index(epoch)]
            else: eval_dict = None
            self.log_fn(epoch, train_dict=train_dict, eval_dict=eval_dict)

    def run(self):
        if self.args.resume: self.resume()
        super(DiffusionExperiment, self).run(epochs=self.args.epochs)

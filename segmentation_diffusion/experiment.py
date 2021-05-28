import torch
from diffusion_utils.loss import elbo_bpd
from diffusion_utils.utils import add_parent_path

add_parent_path(level=2)
from diffusion_utils.experiment import DiffusionExperiment
from diffusion_utils.experiment import add_exp_args as add_exp_args_parent


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        for x in self.train_loader:
            self.optimizer.zero_grad()
            loss = elbo_bpd(self.model, x.to(self.args.device))
            loss.backward()
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()
            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.train_loader:
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
            print('')

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x in self.eval_loader:
                loss = elbo_bpd(self.model, x.to(self.args.device))
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('     Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpd': loss_sum/loss_count}

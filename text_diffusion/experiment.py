import math
import torch
from diffusion_utils.experiment import DiffusionExperiment, add_exp_args


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        loss_moving = None
        for iteration, (x, length) in enumerate(self.train_loader):
            x, length = x.to(self.args.device), length.to(self.args.device)
            num_elem = length.sum()
            loss = - self.model.log_prob(x).sum() / (math.log(2) * num_elem)
            loss.backward()
            if (iteration + 1) % self.args.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)

            if loss_moving is None:
                loss_moving = loss.detach().cpu().item()
            else:
                loss_moving = .99 * loss_moving + .01 * loss.detach().cpu().item()

            if self.args.debug and loss_count > self.args.debug:
                break
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_moving), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpc': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()

        print('sqrt |Lt_history|^2')
        sqrt_Lt = torch.sqrt(self.model.Lt_history)
        print(' '.join(f'{item.item():.2f}' for item in sqrt_Lt))
        print()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x, length in self.eval_loader:
                x, length = x.to(self.args.device), length.to(self.args.device)
                num_elem = length.sum()
                loss = - self.model.log_prob(x).sum() / (math.log(2) * num_elem)
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating train. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')

        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for x, length in self.eval_loader:
                x, length = x.to(self.args.device), length.to(self.args.device)
                num_elem = length.sum()
                loss = - self.model.log_prob(x).sum() / (math.log(2) * num_elem)
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpc': loss_sum/loss_count}

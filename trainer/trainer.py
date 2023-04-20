import numpy as np
import torch

from torchvision.utils import make_grid
from timm.data import Mixup
from time import time

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, is_distributed, rank, trial, valid_data_loader=None, lr_scheduler=None, len_epoch=None, scorer=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, rank, is_distributed=is_distributed, trial=trial, scorer=scorer)
        self.config = config
        self.device = device
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = torch.distributed.get_world_size() if is_distributed else 1
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.mixup_fn = Mixup(**config['mixup']['args'])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        start_time = time()
        iteration_time = start_time

        if self.is_distributed:
            self.data_loader.sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # TODO: delete flag, simply set prob to 0
            if self.config['mixup']['flag']:
                data, target = self.mixup_fn(data, target)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if self.is_distributed:
                torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.reduce(output, dst=0, op=torch.distributed.ReduceOp.SUM)

            if self.rank == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item()/self.world_size)
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output/self.world_size, target))

                if batch_idx % self.log_step == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()
                    self.logger.info('Train Epoch: {} {} Loss: {:.6f} - Elapsed Time: {:.3f} - Avg Iteration time: {:.3f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()/self.world_size,
                        iteration_time - start_time,
                        (iteration_time - previous_iteration_time)/self.log_step))

            if batch_idx == self.len_epoch:
                break

        if self.rank == 0:
            log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            if self.rank == 0:
                log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            # timm scheduler needs epoch
            self.lr_scheduler.step(epoch)
        return log if self.rank == 0 else None

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        start_time = time()

        # TODO: us is_valid_distributed flag
        if self.is_distributed:
            self.valid_data_loader.sampler.set_epoch(epoch)

        if self.rank == 0:
            print("Validating...")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                if self.is_distributed:
                    torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.reduce(output, dst=0, op=torch.distributed.ReduceOp.SUM)

                if self.rank == 0:
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item()/self.world_size)
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output/self.world_size, target, is_logits=False))

                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # TODO: what are we saving down here?
        """if self.rank == 0:
            # add histogram of model parameters to the tensorboard
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')"""
        if self.rank == 0:
            print("Validation Elapsed Time: {:.3f}".format(time() - start_time))
        return self.valid_metrics.result() if self.rank == 0 else None

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

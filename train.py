import argparse
import collections
import os
import optuna

import torch
import torch.distributed as dist
import numpy as np
import data_loader.data_loaders as module_data
import losses.loss as module_loss
import models.metric as module_metric
import models.model_hub as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, build_lr_scheduler


def main(config, trials=None):

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # Hyperparameter optimization
    if trials is not None:
        config['optimizer']['args']['lr'] = trials.suggest_float('lr', 0.0001, 0.01)
        config['optimizer']['args']['weight_decay'] = trials.suggest_float('weight_decay', 0.05, 0.5)
        config['optimizer']['betas'] = [trials.suggest_float('beta1', 0.0, 1.0), trials.suggest_float('beta2', 0.0, 1.0)]

        config['mixup']['args']['mixup_alpha'] = trials.suggest_float('m_alpha', 0.5, 1.0)
        config['mixup']['args']['cutmix_alpha'] = trials.suggest_float('c_alpha', 0.5, 1.0)
        config['mixup']['args']['label_smoothing'] = trials.suggest_float('label_smoothing', 0.0, 0.5)

        config['lr_scheduler']['warmup_epochs'] = trials.suggest_int('warmup_epochs', 3, 8)


    n_gpus = len(config['gpu_list'])
    is_distributed = n_gpus > 1

    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        print("Using multiprocessing...")
        dev_ids = ["cuda:{0}".format(x) for x in config['gpu_list']]
        print("Using DistributedDataParallel on these devices: {}".format(dev_ids))

        score = torch.multiprocessing.spawn(main_worker_function, nprocs=n_gpus, args=(n_gpus, is_distributed, config, trials))
    else:
        print("Not using multiprocessing...")
        score = main_worker_function(0, n_gpus, is_distributed, config, trials)

    return score


def main_worker_function(rank, world_size, is_distributed, config, trials):

    if is_distributed:
        device = config['gpu_list'][rank]
        print("Running main worker function on device: {}".format(device))
        dist.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)
    else:
        device = prepare_device(config['gpu_list'])

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data,
                                  is_distributed=is_distributed, rank=rank, world_size=world_size)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    if rank == 0:
        logger = config.get_logger('train')
        logger.info(model)
    model = model.to(device)
    if is_distributed:
        # If BatchNorm is used, convert it to SyncBatchNorm
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    # TODO: torch.compile() for faster inference HERE, are savable?

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss, rank=rank)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = build_lr_scheduler(config, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      is_distributed=is_distributed,
                      rank=rank,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      trial=trials)

    score = trainer.train()

    if is_distributed:
        dist.destroy_process_group()

    return score


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='train_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    # Start hyperparameter optimization
    # Guide: https://towardsdatascience.com/hyperparameter-tuning-of-neural-networks-with-optuna-and-pytorch-22e179efc837
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=123),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda trial: main(config, trial), n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

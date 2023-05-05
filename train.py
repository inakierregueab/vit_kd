import argparse
import collections
import os

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


def main(config):

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    n_gpus = len(config['gpu_list'])
    is_distributed = n_gpus > 1
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12355 + config['gpu_list'][0])

        print("Using multiprocessing...")
        dev_ids = ["cuda:{0}".format(x) for x in config['gpu_list']]
        print("Using DistributedDataParallel on these devices: {}".format(dev_ids))

        torch.multiprocessing.spawn(main_worker_function, nprocs=n_gpus, args=(n_gpus, is_distributed, config))

    else:
        print("Not using multiprocessing...")
        main_worker_function(0, n_gpus, is_distributed, config)


def main_worker_function(rank, world_size, is_distributed, config):

    if is_distributed:
        device = config['gpu_list'][rank]
        torch.cuda.set_device(device)
        print("Running main worker function on device: {}".format(device))
        dist.init_process_group('nccl', init_method='env://', world_size=world_size, rank=rank)
        # TODO:more info about warning inside DDP
        #os.environ['NCCL_DEBUG'] = 'TRACE'

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
    # TODO: torch.compile() failing

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
                      lr_scheduler=lr_scheduler)

    trainer.train()

    if is_distributed:
        dist.destroy_process_group()


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
    main(config)

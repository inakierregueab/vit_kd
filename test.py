import argparse
import torch
from tqdm import tqdm
import data_loader.test_loader as module_data
import models.metric as module_metric
import models.model_hub as module_arch
from parse_config import ConfigParser
from utils.util import prepare_device


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # load weights from checkpoint
    checkpoint_path = str(config.resume)
    logger.info('Loading checkpoint: {} ...'.format(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)


    # get function handles of loss and metrics
    loss_fn = torch.nn.CrossEntropyLoss()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # prepare model for testing
    device = prepare_device(config['gpu_list'])
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)[0][0]

            #
            # TODO: save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target, is_logits=False) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='test_config.json', type=str,
                      help='config file path (default: None)')
    # TODO: change from docker-compose
    args.add_argument('-r', '--resume', default="./../saved/models/SCE/0520_072107/checkpoint-epoch40.pth", type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

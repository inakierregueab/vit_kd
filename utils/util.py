import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

from timm.scheduler import create_scheduler

from utils.param_store import ParameterStore


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

# TODO: add support for multi-gpu training
def prepare_device(gpu_list):

    if torch.cuda.is_available():
        print("Using a single GPU...")
        device = torch.device(f'cuda:{gpu_list[0]}')

    elif torch.backends.mps.is_available():
        print("MPS is available on this machine, training will be performed on MPS.")
        device = torch.device('mps')

    else:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device = torch.device('cpu')

    return device

def build_lr_scheduler(config, optimizer):
    lr_config = config['lr_scheduler']
    lr_config['epochs'] = config['trainer']['epochs']
    lr_parsed_config = ParameterStore(lr_config)
    lr_scheduler, _ = create_scheduler(lr_parsed_config, optimizer)
    return lr_scheduler


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)



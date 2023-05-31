import os

import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, DistributedSampler, Subset
from torch.utils.data.dataloader import default_collate

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from data_loader.ra_sampler import RASampler


class IMNETDataLoader(DataLoader):
    """
    ImageNet dataloader, with support for transforms and validation split
    """
    def __init__(self,
                 data_dir,
                 batch_size,
                 is_distributed,
                 rank,
                 world_size,
                 repeated_aug=False,
                 num_workers=0,
                 transform_config=None,
                 collate_fn=default_collate,
                 pin_memory=True,
                 persistent_workers=True):

        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')

        self.train_transform = self.get_transforms(transform_config, is_train=True)
        self.val_transform = self.get_transforms(transform_config, is_train=False)

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)

        # Subset for debugging
        # define the size of your subset
        #train_subset_size = 100000
        val_subset_size = 10000
        # get the indices of all images in the full dataset
        #train_indices = list(range(len(self.train_dataset)))
        val_indices = list(range(len(self.val_dataset)))
        # randomly choose a subset of the indices
        #train_subset_indices = np.random.choice(train_indices, train_subset_size, replace=False)
        val_subset_indices = np.random.choice(val_indices, val_subset_size, replace=False)
        # Subset for distributed training
        if is_distributed:
            #torch.distributed.broadcast(torch.tensor(train_subset_indices, device=rank), 0)
            torch.distributed.broadcast(torch.tensor(val_subset_indices, device=rank), 0)
            torch.distributed.barrier()

        # generate subset
        #self.train_sub_dataset = Subset(self.train_dataset, train_subset_indices)
        self.val_sub_dataset = Subset(self.train_dataset, val_subset_indices)

        # TODO: delete subset when full training
        if is_distributed:
            if repeated_aug:
                self.train_sampler = RASampler(self.train_sub_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            else:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank,
                                                        shuffle=True, drop_last=False)

            # TODO: add flag for distributed validation
            if len(self.val_dataset) % world_size != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            # TODO: drop last and shuffle for better graphics
            self.valid_sampler = DistributedSampler(self.val_sub_dataset, num_replicas=world_size, rank=rank,
                                                    shuffle=True, drop_last=True)
        else:
            self.train_sampler = RandomSampler(self.train_sub_dataset)
            self.valid_sampler = SequentialSampler(self.val_sub_dataset)

        self.init_kwargs = {
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
        }
        super().__init__(dataset=self.train_dataset,sampler=self.train_sampler, drop_last=True,**self.init_kwargs)

    def get_transforms(self, transform_config, is_train):
        resize_im = transform_config['input_size'] > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=transform_config['input_size'],
                is_training=True,
                color_jitter=transform_config['color_jitter'],
                auto_augment=transform_config['aa'],
                interpolation=transform_config['train_interpolation'],
                re_prob=transform_config['reprob'],
                re_mode=transform_config['remode'],
                re_count=transform_config['recount'],
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(
                    transform_config['input_size'], padding=4)
            return transform

        t = []
        if resize_im:
            size = int((256/224)*transform_config['input_size'])
            t.append(
                transforms.Resize(size),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(transform_config['input_size']))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def split_validation(self):
        return DataLoader(dataset=self.val_sub_dataset, sampler=self.valid_sampler, drop_last=True,  **self.init_kwargs)






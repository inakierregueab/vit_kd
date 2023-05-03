import os
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

        self.train_dir = os.path.join(data_dir, 'val')
        self.val_dir = os.path.join(data_dir, 'val')

        self.train_transform = self.get_transforms(transform_config, is_train=True)
        self.val_transform = self.get_transforms(transform_config, is_train=False)

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)

        # Subset for debugging
        train_indices = torch.arange(20000)
        val_indices = torch.arange(5000)
        #self.train_dataset = Subset(self.train_dataset, train_indices)
        self.val_dataset = Subset(self.train_dataset, val_indices)

        if is_distributed:
            if repeated_aug:
                self.train_sampler = RASampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            else:
                self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank,
                                                        shuffle=True, drop_last=False)

            # TODO: add flag for distributed validation
            if len(self.val_dataset) % world_size != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            self.valid_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank,
                                                    shuffle=False, drop_last=False)
        else:
            self.train_sampler = RandomSampler(self.train_dataset)
            self.valid_sampler = SequentialSampler(self.val_dataset)

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
        return DataLoader(dataset=self.val_dataset, sampler=self.valid_sampler, drop_last=False,  **self.init_kwargs)






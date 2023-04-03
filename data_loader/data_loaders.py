import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


class IMNETDataLoader(DataLoader):
    """
    ImageNet dataloader, with support for transforms and validation split
    """
    def __init__(self,
                 data_dir,
                 batch_size,
                 num_workers=1,
                 transform_config=None,
                 collate_fn=default_collate,
                 pin_memory=True):

        # TODO: charge train to 'train'
        self.train_dir = os.path.join(data_dir, 'val')
        self.val_dir = os.path.join(data_dir, 'val')

        self.train_transform = self.get_transforms(transform_config, is_train=True)
        self.val_transform = self.get_transforms(transform_config, is_train=False)

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_transform)

        self.train_sampler = RandomSampler(self.train_dataset)
        self.valid_sampler = SequentialSampler(self.val_dataset)
        #self.train_sampler = RandomSampler(self.train_dataset, num_samples=128)
        #self.valid_sampler = RandomSampler(self.val_dataset, num_samples=128)

        self.init_kwargs = {
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }

        super().__init__(dataset=self.train_dataset,sampler=self.train_sampler, drop_last=True, **self.init_kwargs)

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
            # TODO: define interpolation mode
            t.append(
                transforms.Resize(size),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(transform_config['input_size']))

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def split_validation(self):
        return DataLoader(dataset=self.val_dataset, sampler=self.valid_sampler, drop_last=False,  **self.init_kwargs)






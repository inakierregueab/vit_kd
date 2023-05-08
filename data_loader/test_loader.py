import os

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class TestLoader(DataLoader):
    def __init__(self,
                 data_dir,
                 batch_size,
                 input_size=224,
                 num_workers=4,
                 collate_fn=None,
                 pin_memory=True,
                 persistent_workers=True):

        self.input_size = input_size

        self.test_dir = os.path.join(data_dir, 'val')

        self.test_transform = self.get_transforms()

        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.test_transform)

        self.sampler = None  #SubsetRandomSampler(list(range(1000)))

        super().__init__(self.test_dataset,
                         batch_size=batch_size,
                         sampler=self.sampler,
                         shuffle=False,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,
                         persistent_workers=persistent_workers)

    def get_transforms(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        return transform
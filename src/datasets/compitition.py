import os
import torch
import torchvision
import wilds

from wilds.common.data_loaders import get_train_loader, get_eval_loader


class Compitition:
    test_subset = None

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=4,
                 subset='test',
                 classnames=None,
                 **kwargs):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_location = os.path.join(location, 'compitition', 'train')
        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.train_location, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        self.test_location = os.path.join(location, 'compitition',
                                          self.test_subset)
        print("Loading Test Data from ", self.test_location)
        self.test_dataset = torchvision.datasets.ImageFolder(
            root=self.test_location, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)


class CompititionVal(Compitition):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'val'
        super().__init__(*args, **kwargs)


class CompititionTest(Compitition):
    def __init__(self, *args, **kwargs):
        self.test_subset = 'test'
        super().__init__(*args, **kwargs)
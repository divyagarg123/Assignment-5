from __future__ import print_function
import torch
from torchvision import datasets, transforms
import utils as ut

class DataLoader():
    def __init__(self):
        self.cuda = ut.check_for_cuda()

    def transforms(self):
        train_transforms = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize((0.1307), (0.3081))
                                               ])
        test_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.1307), (0.3081))
                                                ])
        return train_transforms, test_transforms

    def load_dataset(self):
        train_transforms, test_transforms = self.transforms()
        train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
        test = datasets.MNIST('./data', train=True, download=True, transform=test_transforms)
        return train, test

    def return_loaders(self):
        train, test = self.load_dataset()
        dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if self.cuda else dict(shuffle=True,
                                                                                                           batch_size=64)
        train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
        return train_loader, test_loader

import torch
from torchvision import datasets, transforms
from datasets.dataset import Dataset
from os import path


class MNIST(Dataset):
    def __init__(self, batch_size, data_location=path.join(path.dirname(__file__), "../../data"), download=False):
        super().__init__(batch_size)
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_location, train=True, download=download,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_location, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)
        self.mask_value = -0.4242

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

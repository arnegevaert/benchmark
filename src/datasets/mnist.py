import torch
from torchvision import datasets, transforms
from datasets.image_dataset import ImageDataset
from os import path


class MNIST(ImageDataset):
    def __init__(self, batch_size, data_location=path.join(path.dirname(__file__), "../../data"),
                 download=False, shuffle=True):
        super().__init__(batch_size)
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_location, train=True, download=download,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_location, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=shuffle)
        self.mask_value = -0.4242
        self.sample_shape = (1, 28, 28)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_sample_shape(self):
        return self.sample_shape


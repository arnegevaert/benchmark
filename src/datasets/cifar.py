from datasets import Dataset
import torch
from torchvision import datasets, transforms
from os import path


class Cifar(Dataset):
    def __init__(self, batch_size, data_location=path.join(path.dirname(__file__), "../../data"),
                 download=False, shuffle=True, version="cifar10"):
        super().__init__(batch_size, [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if version not in ["cifar10", "cifar100"]:
            raise ValueError("version must be in {cifar10, cifar100}")
        transform = transforms.Compose(self.transforms)
        ds_constructors = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
        self.train_loader = torch.utils.data.DataLoader(
            ds_constructors[version](data_location, train=True, download=download, transform=transform),
            batch_size=batch_size, shuffle=shuffle)
        self.test_loader = torch.utils.data.DataLoader(
            ds_constructors[version](data_location, train=False, download=download, transform=transform),
            batch_size=batch_size, shuffle=shuffle)
        self.mask_value = 0  # TODO verify if this is ok
        self.sample_shape = (3, 32, 32)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader

    def get_sample_shape(self):
        return self.sample_shape

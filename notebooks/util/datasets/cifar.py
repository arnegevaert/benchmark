from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Cifar:
    def __init__(self, batch_size, data_location, download=False, shuffle=True, version="cifar10"):
        self.sample_shape = (3, 32, 32)
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ds_constructors = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
        self.train_loader = DataLoader(
            ds_constructors[version](data_location, train=True, download=download, transform=transform),
            batch_size=batch_size, shuffle=shuffle, drop_last=True)
        self.test_loader = DataLoader(
            ds_constructors[version](data_location, train=False, download=download, transform=transform),
            batch_size=batch_size, shuffle=shuffle, drop_last=True)
        self.mask_value = -0.4242

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader

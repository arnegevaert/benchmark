from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch


class Cifar(Dataset):
    def __init__(self, data_location, train, download=False, version="cifar10"):
        self.sample_shape = (3, 32, 32)
        mean = (0.4914, 0.4821, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.denormalization = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
        self.mask_value = 0.
        ds_constructors = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
        self.dataset = ds_constructors[version](data_location, train, self.transform, download=download)
        self.num_classes = 10 if version == "cifar10" else 100

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def denormalize(self, item):
        return torch.stack([self.denormalization(item[i]) for i in range(item.shape[0])], dim=0)
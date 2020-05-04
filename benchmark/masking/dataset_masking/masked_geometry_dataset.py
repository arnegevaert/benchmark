import torch
from torchvision import transforms, datasets
from os import path
import os


class MaskedGeometryDataset:
    def __init__(self, batch_size, data_location, shuffle=True):
        self.batch_size = batch_size
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])  # TODO normalization
        self.train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=path.join(data_location, "train"),
                                 transform=data_transform), batch_size=batch_size, shuffle=shuffle)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(root=path.join(data_location, "test"),
                                 transform=data_transform), batch_size=batch_size, shuffle=shuffle)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader


def generate_masked_geometry_dataset(location: str, width: int, height: int, train_size: int, test_size: int):
    os.makedirs(path.join(location, "train"))
    os.makedirs(path.join(location, "test"))

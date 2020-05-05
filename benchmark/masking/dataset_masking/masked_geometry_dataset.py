import torch
from torchvision import transforms
from os import path
import os


class MaskedGeometryDataset:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_location, train=True, include_masks=False):
            self.data_transform = transforms.Compose([
                transforms.ToTensor()
            ])  # TODO normalization
            self.data_location = path.join(data_location, "train" if train else "test")
            self.include_masks = include_masks

        def __len__(self):
            pass

        def __getitem__(self, item):
            pass

    def __init__(self, batch_size, data_location):
        self.batch_size = batch_size
        self.data_location = data_location

    def get_dataloader(self, train=True, include_masks=False, shuffle=True):
        return torch.utils.data.DataLoader(
            MaskedGeometryDataset.Dataset(self.data_location, train, include_masks),
            batch_size=self.batch_size, shuffle=shuffle)


def generate_masked_geometry_dataset(location: str, width: int, height: int, train_size: int, test_size: int):
    os.makedirs(path.join(location, "train"))
    os.makedirs(path.join(location, "test"))

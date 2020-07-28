from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Cifar:
    def __init__(self, batch_size, data_location, download=False, shuffle=True, version="cifar10"):
        self.sample_shape = (3, 32, 32)
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.ds_constructors = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
        self.data_location = data_location
        self.version = version
        self.download = download
        self.shuffle = shuffle
        self.mask_value = -1
        self.num_classes = 10 if version == "cifar10" else 100

    def get_dataloader(self, train=True):
        return DataLoader(
            self.ds_constructors[self.version](self.data_location, train=train,
                                               download=self.download, transform=self.transform),
            batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=4)

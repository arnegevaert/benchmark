from torchvision import datasets, transforms
from util.datasets import TrainableDataset
from torch.utils.data import DataLoader


class MNIST(TrainableDataset):
    def __init__(self, batch_size, data_location, download=False, shuffle=True):
        super().__init__(batch_size, (1, 28, 28))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_loader = DataLoader(
            datasets.MNIST(data_location, train=True, download=download, transform=transform),
            batch_size=batch_size, shuffle=shuffle)

        self.test_loader = DataLoader(
            datasets.MNIST(data_location, train=False, transform=transform),
            batch_size=batch_size, shuffle=shuffle)
        self.mask_value = -0.4242

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_sample_shape(self):
        return self.sample_shape

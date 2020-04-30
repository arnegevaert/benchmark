from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNIST:
    def __init__(self, batch_size, data_location, download=False, shuffle=True):
        self.sample_shape = (1, 28, 28)
        self.batch_size = batch_size
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

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader

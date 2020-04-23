from util.datasets import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNIST(Dataset):
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

    def get_sample_shape(self):
        return self.sample_shape

    def get_train_data(self):
        for samples, labels in iter(self.train_loader):
            yield samples.detach().numpy(), labels.detach().numpy()

    def get_test_data(self):
        for samples, labels in iter(self.test_loader):
            yield samples.detach().numpy(), labels.detach().numpy()

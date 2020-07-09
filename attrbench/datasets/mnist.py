from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNIST:
    def __init__(self, batch_size, data_location, download=False, shuffle=True):
        self.sample_shape = (1, 28, 28)
        self.batch_size = batch_size
        self.data_location = data_location
        self.download = download
        self.shuffle = shuffle
        self.mask_value = -0.4242
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.num_classes = 10

    def get_dataloader(self, train=True):
        return DataLoader(
            datasets.MNIST(self.data_location, train=train, transform=self.transform),
            batch_size=self.batch_size, shuffle=self.shuffle)

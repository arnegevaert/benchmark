from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, data_location, train, download=False):
        self.sample_shape = (1, 28, 28)
        self.mean = 0.1307
        self.std = 0.3081
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])
        self.num_classes = 10
        self.dataset = datasets.MNIST(data_location, train=train, transform=self.transform, download=download)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def denormalize(self, item):
        return item * self.std + self.mean
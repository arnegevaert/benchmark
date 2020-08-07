from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, data_location, train, download=False):
        self.sample_shape = (1, 28, 28)
        # Minimum value in normalized dataset represents black
        # This is not the dataset average, but it is the background color,
        # which is a better "neutral" value than the dataset mean (which is grey)
        self.mask_value = -0.4242
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.num_classes = 10
        self.dataset = datasets.MNIST(data_location, train=train, transform=self.transform, download=download)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

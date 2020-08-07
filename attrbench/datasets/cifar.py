from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Cifar:
    def __init__(self, batch_size, data_location, download=False, shuffle=True, version="cifar10"):
        self.sample_shape = (3, 32, 32)
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.ds_constructors = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}
        self.data_location = data_location
        self.version = version
        self.download = download
        self.shuffle = shuffle
        self.mask_value = 0.
        self.num_classes = 10 if version == "cifar10" else 100

    def get_dataloader(self, train=True):
        return DataLoader(
            self.ds_constructors[self.version](self.data_location, train=train,
                                               download=self.download, transform=self.transform),
            batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True, num_workers=4)


if __name__ == "__main__":
    from tqdm import tqdm
    import torch

    cifar = Cifar(batch_size=64, data_location="../../data/CIFAR10", shuffle=False)
    dl = cifar.get_dataloader(train=False)
    batch_means = []
    batch_sdevs = []
    for batch, labels in tqdm(dl):
        batch_means.append(torch.mean(batch.flatten(-2), dim=-1))
        batch_sdevs.append(torch.std(batch.flatten(-2), dim=-1))
    means = torch.cat(batch_means, dim=0)
    sdevs = torch.cat(batch_sdevs, dim=0)
    mean = torch.mean(means, dim=0)
    sd = torch.mean(sdevs, dim=0)
    print(mean)
    print(sd)

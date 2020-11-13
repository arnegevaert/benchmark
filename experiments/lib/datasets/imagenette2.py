from torchvision import datasets, transforms
from torch.utils.data import Dataset
from os import path
import torch


class ImageNette(Dataset):
    def __init__(self, data_location, train, image_size=224):
        self.mask_value = 0
        self.sample_shape = (3, image_size, image_size)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.denormalization = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
        tf = train_transforms if train else val_transforms
        version = "train" if train else "val"
        self.dataset = datasets.ImageFolder(path.join(data_location, version), tf)
        self.num_classes = 10

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def denormalize(self, item):
        return torch.stack([self.denormalization(item[i]) for i in range(item.shape[0])], dim=0)

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from os import path


class ImageNette(Dataset):
    def __init__(self, data_location, train, image_size=224):
        self.mask_value = 0
        self.sample_shape = (3, image_size, image_size)
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tf = train_transforms if train else val_transforms
        version = "train" if train else "val"
        self.dataset = datasets.ImageFolder(path.join(data_location, version), tf)
        self.num_classes = 10

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from os import path


class ImageNette:
    def __init__(self, batch_size, data_location, shuffle=True, image_size=224):
        self.sample_shape = (3, image_size, image_size)
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train_dataset = datasets.ImageFolder(path.join(data_location, 'train'), self.train_transforms)
        self.val_dataset = datasets.ImageFolder(path.join(data_location, 'val'), self.val_transforms)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
        self.test_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader

    def get_dataloader(self, train=True):
        return self.get_train_data() if train else self.get_test_data()

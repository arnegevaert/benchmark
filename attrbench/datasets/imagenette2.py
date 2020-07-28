from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from os import path


class ImageNette:
    def __init__(self, batch_size, data_location, shuffle=True, image_size=224):
        self.mask_value = 0
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
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = 10

    def get_dataloader(self, train=True):
        return DataLoader(self.train_dataset if train else self.val_dataset,
                          batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)

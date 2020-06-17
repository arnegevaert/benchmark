import torch
from torchvision import transforms
from os import path
from skimage import io
import os


class MaskedGeometryDataset:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data_location, train=True, include_masks=False):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5))  # Move features from [0, 1] to [-1, 1]
            ])
            self.mask_transform = transforms.ToTensor()
            data_location = path.join(data_location, "train" if train else "test")
            self.img_location = path.join(data_location, "imgs")
            self.mask_location = path.join(data_location, "masks")
            self.files = [name for name in sorted(os.listdir(self.img_location))
                          if path.isfile(path.join(self.img_location, name))]
            self.include_masks = include_masks
            self.classes = {"circle": 0, "square": 1}

        def __len__(self):
            return len(self.files)

        def __getitem__(self, item):
            filename = self.files[item]
            label = int(filename[0])
            image = self.transform(io.imread(path.join(self.img_location, filename)))
            if self.include_masks:
                mask = self.mask_transform(io.imread(path.join(self.mask_location, filename))).squeeze()
                return image, torch.tensor(label), mask
            return image, torch.tensor(label)

    def __init__(self, batch_size, data_location):
        self.batch_size = batch_size
        self.data_location = data_location

    def get_dataloader(self, train=True, include_masks=False, shuffle=True):
        return torch.utils.data.DataLoader(
            MaskedGeometryDataset.Dataset(self.data_location, train, include_masks),
            batch_size=self.batch_size, shuffle=shuffle)

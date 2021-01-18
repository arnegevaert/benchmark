from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import imgaug.augmenters as iaa

class DropoutDataset(Dataset):
    ########################################
    # takes a dataset and sets between
    # 0 and 90% of pixels randomly to zero
    ########################################
    def __init__(self, dataset, aug = None):
        self.dataset = dataset
        self.aug = aug if aug else iaa.Sometimes(0.95, iaa.Dropout(p=(0.0, 0.90)))

    def __getitem__(self, item):
        im, label = self.dataset[item]
        im = im.transpose(0, 2).numpy()
        im = self.aug.augment_image(im)
        im = torch.from_numpy(im).transpose(0,2)
        return im, label

    def __len__(self):
        return len(self.dataset)
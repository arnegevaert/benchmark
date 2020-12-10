import h5py
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from os import path
import torchvision.transforms.functional as TF
import random
import pandas as pd
import numpy as np
from pydicom import dcmread
import torch
import matplotlib.pyplot as plt



class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

class PcamDataset(Dataset):
    def __init__(self, data_loc, train, use_val_set=False):
        super().__init__()

        transfList = [
            transforms.ToTensor(),
        ]

        if train:
            self.file_path_x = path.join(data_loc,'camelyonpatch_level_2_split_train_x.h5')
            self.file_path_y = path.join(data_loc, 'camelyonpatch_level_2_split_train_y.h5')
            transfList.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRotationTransform(angles=[-90, 0, 90, 180])
            ])
        elif use_val_set:
            self.file_path_x = path.join(data_loc,'camelyonpatch_level_2_split_valid_x.h5')
            self.file_path_y = path.join(data_loc, 'camelyonpatch_level_2_split_valid_y.h5')
        else:
            self.file_path_x = path.join(data_loc,'camelyonpatch_level_2_split_test_x.h5')
            self.file_path_y = path.join(data_loc, 'camelyonpatch_level_2_split_test_y.h5')
        self.file_path = data_loc
        self.dataset = None
        with h5py.File(self.file_path_y, 'r') as file:
            self.dataset_len = len(file["y"])

        self.transform = transforms.Compose(transfList)
        self.dataset_x = None
        self.dataset_y = None

    def __getitem__(self, index):
        if not self.dataset_x: # cant set dataset in init, opened h5py files can't be serialised -> can't use multiple
                    #processes
            self.dataset_x = h5py.File(self.file_path_x, 'r')["x"]
            self.dataset_y = h5py.File(self.file_path_y, 'r')["y"]

        image = self.dataset_x[index]
        label = self.dataset_y[index].squeeze()
        # image= np.moveaxis(image, 2, 0) /255
        image = image /255
        image  =self.transform(image)

        return image, label.astype(np.int64)

    def __len__(self):
        return self.dataset_len


if __name__ == '__main__':
    dataloc= "D:\Project\Data\PCAM"

    ds = PcamDataset(dataloc,train=True)
    dl = DataLoader(ds, batch_size=512, num_workers=0)
    for batch, label in dl:
        print("debug")
        break

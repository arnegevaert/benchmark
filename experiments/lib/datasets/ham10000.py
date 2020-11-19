import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from os import path

import random
import pandas as pd
import numpy as np
from pydicom import dcmread
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class HAM10000(Dataset):
    def __init__(self, data_loc, train, imsize=300):
        super().__init__()

        transfList = [
            transforms.ToTensor(),
            transforms.CenterCrop(450),
            transforms.Resize(imsize)
        ]

        self.mean = np.array([0.7635, 0.5376, 0.5608])
        self.std = np.array([0.0997, 0.1298, 0.1454])

        self.class_weights = np.array([ 9.00224719,  1.49366145, 19.49391727, 30.58015267,  9.1149033 ,
       87.08695652, 70.28070175])

        if train:
            csv_file = "train.csv"
            transfList.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRotationTransform(angles=[-90, 0, 90, 180]),
                transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            ])
        else:
            csv_file = "test.csv"
        transfList.append(transforms.Normalize(torch.from_numpy(self.mean).float(),
                                     torch.from_numpy(self.std).float()))
        self.transform = transforms.Compose(transfList)
        df = pd.read_csv(path.join(data_loc,csv_file))
        images = df["image"].to_list()
        self.images = [path.join(data_loc,"Images", im+'.jpg') for im in images ]
        self.labels = df[["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]].to_numpy()

    def __getitem__(self, item):
        im_path = self.images[item]
        label = self.labels[item]

        im = np.asarray(Image.open(im_path))
        im = im.copy() # to suppress a warning related to using PIL and torchvision transforms
        im = self.transform(im)
        return im, label.argmax()

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    my_ds = HAM10000('D:\Project\Data\HAM10000',train=True,imsize=224)
    # my_ds.transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.CenterCrop(450),
    #     transforms.Resize(300)
    #         ])
    dl = DataLoader(my_ds, batch_size=1024, num_workers=4)
    n_images = len(my_ds)
    mean = torch.tensor([0.,0.,0.])
    var = torch.tensor([0.,0.,0.])
    i = 0
    for batch, label in dl:
        mean += batch.mean((2,3)).sum(0) / n_images #
        var += batch.var((2,3)).sum(0) / n_images

    std = torch.sqrt(var)
    print(mean)
    print(std)
    print(i)
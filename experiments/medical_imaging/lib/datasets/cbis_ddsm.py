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
from PIL import Image, ImageOps
import cv2 as cv

##########################
#for testing:
original_crops= False
##########################

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class CBIS_DDSM_patches(Dataset):
    def __init__(self, data_loc, train, imsize):
        super(CBIS_DDSM_patches,self).__init__()

        if train:
            df = pd.read_csv(path.join(data_loc,"mass_case_description_train_set.csv"),usecols=["cropped image file path","pathology"])
        else:
            df = pd.read_csv(path.join(data_loc, "mass_case_description_test_set.csv"))
        self.df = df
        labels = [pat =="MALIGNANT" for pat in df['pathology']]
        self.labels = np.array(labels,dtype=np.int64)
        # self.labels = np.zeros((len(labels),2)) # one-hot
        # self.labels[np.arange(len(labels)), labels] = 1

        if original_crops:
            name = 'crop.dcm'
            self.mean = 47541.1914
            self.std = 7755.1230
        else:
            name = 'my_ROI.png'
            self.mean = 0.4556
            self.std = 0.1850

        self.paths = [path.join(data_loc,"CBIS-DDSM",path.dirname(loc),name)for loc in df["cropped image file path"]]


        if train:
            self.transforms =transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),
                transforms.Normalize(mean=self.mean, std=self.std),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                MyRotationTransform(angles=[-90, 0, 90, 180]),
                # transforms.RandomAffine(degrees=15,translate=(0.05,0.05),scale=(0.9,1.1),shear=5, fillcolor=None)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((imsize, imsize)),

                transforms.Normalize(mean=self.mean,std=self.std)
            ])

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        if original_crops:
            im = dcmread(path)
            image = im.pixel_array.astype(np.float32)
        else:
            i = cv.imread(path, cv.IMREAD_ANYDEPTH)
            i = (i - i.min()) / (i.max() - i.min())
            image = i.astype(np.float32)

        image = self.transforms(image)
        image = image.expand(3,*image.shape[1:])
        return image, label

    def __len__(self):
        return len(self.paths)


def imshow(img, title):
    npimg = torchvision.utils.make_grid(img).cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # calc std and mean
    my_ds = CBIS_DDSM_patches('D:\Project\Data\CBIS-DDSM',train=True,imsize = 224)
    my_ds.transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
            ])
    dl = DataLoader(my_ds,batch_size=32,num_workers=4)
    n_images = len(my_ds)
    mean =0.
    var =0.
    i = 0
    for batch, label in dl:
        batch = batch.view(batch.shape[0],-1)
        mean += batch.mean(1).sum() / n_images
        var += batch.var(1).sum() / n_images
        i+=1
    std = torch.sqrt(var)
    print(mean)
    print(std)
    print(i)


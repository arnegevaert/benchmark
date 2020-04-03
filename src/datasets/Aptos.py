from datasets import Dataset
from torchvision import datasets, transforms
from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
from sklearn.utils import class_weight
import pandas as pd


class _IaaWrapper:
    """ imgaug transformations work on batches of images (B,W,H,C). this class simply calls
    the augmentations on a single image (1,W,H,C) """

    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, *args, **kwargs):
        x = x[None]
        x = self.aug(images=x)
        return x[-1]


class MyNumpyDataset(torch.utils.data.Dataset):
    """" creates a torch dataset that returns tensors from numpy arrays of images,
     and applies transform function. transforms are applied per image"""

    def __init__(self, data, labels, transforms=None):
        assert type(data) is np.ndarray
        assert type(labels) is np.ndarray or type(labels) is list
        labels = np.array(labels) if type(labels) is list else labels
        assert data.shape[0] == labels.shape[0]
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


def get_aptos_dataset(data_loc, imsize=224, batch_size=8):
    # get numpy array of aptos images and labels and returns torch DataLoader for train and test set
    Train = pd.read_csv(path.join(data_loc, 'APTOS/train_aptos_2019.csv'))
    Train['id_code'] = Train['id_code'] + '.png'

    Test = pd.read_csv(path.join(data_loc, 'APTOS/test_aptos_2019.csv'))
    Test['id_code'] = Test['id_code'] + '.png'

    x_train_path = path.join(data_loc, 'APTOS/Train_aptos_2019_' + str(imsize) + '.npy')
    if not path.exists(x_train_path):
        raise FileNotFoundError('file ' + x_train_path +
                                ' does not exist. Please create the file or use a different image size')
    x_train = np.load(path.join(data_loc, 'APTOS/Train_aptos_2019_' + str(imsize) + '.npy'))
    x_test = np.load(path.join(data_loc, 'APTOS/Test_aptos_2019_' + str(imsize) + '.npy'))

    y_train = Train['diagnosis'].tolist()
    y_test = Test['diagnosis'].tolist()
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(Train['diagnosis'].tolist()),
        Train['diagnosis'].tolist())
    class_weights = torch.FloatTensor(class_weights)

    aug = iaa.Sequential([
        iaa.Affine(rotate=(-30, 30), scale=(0.95, 1.25), translate_percent=(-0.2, 0.2), shear=(-5, 5), mode="constant",
                   cval=127),
        iaa.Fliplr(0.5),
        iaa.Lambda(lambda x, *args: np.moveaxis(x, 3, 1) / 255)
    ])

    transform = _IaaWrapper(aug)

    ds_train = MyNumpyDataset(x_train, y_train, transforms=transform)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ds_val = MyNumpyDataset(x_test, y_test,
                            transforms=_IaaWrapper(iaa.Lambda(lambda x, *args: np.moveaxis(x, 3, 1) / 255)))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return dl_train, dl_val, class_weights


class Aptos(Dataset):
    def __init__(self, batch_size, img_size=224, data_location=path.join(path.dirname(__file__), "../../data"),
                 shuffle=True, download=False):
        super().__init__(batch_size)

        self.train_loader, self.test_loader, self.class_weights = \
            get_aptos_dataset(imsize=img_size, batch_size=batch_size, data_loc=data_location)
        self.sample_shape = (3, img_size, img_size)
        self.mask_value = 127/255

    def get_train_data(self):
        return self.train_loader

    def get_test_data(self):
        return self.test_loader

    def get_sample_shape(self):
        return self.sample_shape

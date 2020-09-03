from os import path
import numpy as np
import torch
from imgaug import augmenters as iaa
from sklearn.utils import class_weight
import pandas as pd
from torch.utils.data import Dataset


def _move_axis_lambda(x, *args):
    # for use in transformation. python cant pickle lambda functions, so named function is needed
    # if we want dataloader to use multiprocessing
    return np.moveaxis(x, 3, 1) / 255


def _get_aptos_dataset(data_loc, imsize=224, train=True):
    csv_file = "train_aptos_2019.csv" if train else "test_aptos_2019.csv"
    df = pd.read_csv(path.join(data_loc, csv_file))
    df['id_code'] = df['id_code'] + '.png'

    npy_file = f"Train_aptos_2019_{imsize}.npy" if train else f"Test_aptos_2019_{imsize}.npy"
    npy_path = path.join(data_loc, npy_file)
    if not path.exists(npy_path):
        raise FileNotFoundError(f"File {npy_path} does not exist. Please create the file or use a different image size")
    X = np.load(npy_path)
    y = df['diagnosis'].tolist()
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    class_weights = np.array(class_weights, dtype=np.float)
    return X, y, class_weights


class Aptos(Dataset):
    def __init__(self, data_location, img_size=224, train=True):
        self.x_array, self.y_array, self.class_weights = _get_aptos_dataset(imsize=img_size, data_loc=data_location)
        self.sample_shape = (3, img_size, img_size)
        self.mask_value = 127/255

        if train:
            self.transform = iaa.Sequential([
                iaa.Affine(rotate=(-30, 30), scale=(0.95, 1.25), translate_percent=(-0.1, 0.1), shear=(-5, 5), mode="constant",
                           cval=127),
                iaa.Fliplr(0.5),
                iaa.Lambda(_move_axis_lambda)
            ])
        else:
            self.transform = iaa.Lambda(_move_axis_lambda)
        self.num_classes = 5

    def __getitem__(self, item):
        sample = self.transform(images=np.expand_dims(self.x_array[item], axis=0))
        label = self.y_array[item]
        return torch.tensor(sample,dtype=torch.float).squeeze(), torch.tensor(label)

    def __len__(self):
        return self.x_array.shape[0]
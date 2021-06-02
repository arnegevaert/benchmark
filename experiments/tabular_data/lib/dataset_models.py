from torchvision import datasets, transforms
import os
from os import path
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

_DATA_LOC = os.environ["BM_DATA_LOC"] if "BM_DATA_LOC" in os.environ else path.join(path.dirname(__file__), "../../../data")


def get_dataset_model(name, model_name=None, train=False):
    if name== "SeizeIT1":
        with open(path.join(_DATA_LOC,name,"scalar"), 'rb') as f:
            scalar = pickle.load(f)
        with open(path.join(_DATA_LOC,name,"val_dicts"), 'rb') as f:
            val_labels, val_feats= pickle.load(f)
        x_val, y_val = np.concatenate([val_feats[key] for key in val_feats]), np.concatenate(
            [val_labels[key] for key in val_labels])
        x_val = scalar.transform(x_val)
        x_val, y_val = torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64)
        ds = torch.utils.data.TensorDataset(x_val,y_val)
        if model_name == "model_binary":
            model = nn.Sequential(
                nn.Linear(67, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.ReLU(),
                nn.Linear(20, 1))
            model.load_state_dict(torch.load(path.join(_DATA_LOC, "models", name, "model_binary.pt")))
        else:
            model = nn.Sequential(
                nn.Linear(67,20),
                nn.ReLU(),
                nn.Linear(20,20),
                nn.ReLU(),
                nn.Linear(20,2))
            model.load_state_dict(torch.load(path.join(_DATA_LOC,"models",name,"model2.pt")))
        patch_folder = None

    return ds, model, patch_folder


class BasicCNN(nn.Module):
    """
    Basic convolutional network for MNIST
    """
    def __init__(self, num_classes, params_loc=None):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if params_loc:
            # map_location allows taking a model trained on GPU and loading it on CPU
            # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        if x.dtype != torch.float32:
            x = x.float()

        relu = nn.ReLU()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def get_last_conv_layer(self):
        return self.conv2

if __name__ == '__main__':
    ds, m, patch = get_dataset_model("SeizeIT1")
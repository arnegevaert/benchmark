from torchvision import datasets, transforms
import os
from os import path
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from experiments.general_imaging.models import Resnet20


_DATA_LOC = os.environ["BM_DATA_LOC"] if "BM_DATA_LOC" in os.environ else path.join(path.dirname(__file__), "../../data")


def get_dataset_model(name):
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = datasets.MNIST(path.join(_DATA_LOC, "MNIST"), train=False, transform=transform, download=True)
        model = BasicCNN(10, path.join(_DATA_LOC, "models/MNIST/cnn.pt"))
        sample_shape = (28, 28)
    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4821, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        ds = datasets.CIFAR10(path.join(_DATA_LOC, "CIFAR10"), train=False, transform=transform, download=True)
        model = Resnet20(10,path.join(_DATA_LOC, "models/CIFAR10/resnet20.pt"))
        sample_shape = (32, 32)
    elif name == "ImageNette":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        ds = datasets.ImageFolder(path.join(_DATA_LOC, "imagenette2", "val"), transform=transform)
        model = Resnet18(path.join(_DATA_LOC, "models/ImageNette/resnet18.pt"))
        sample_shape = (224, 224)
    else:
        raise ValueError(f"Invalid dataset: {name}")
    return ds, model, sample_shape


class Resnet18(nn.Module):
    """
    Wrapper class around torchvision Resnet18 model,
    with 10 output classes and function to get the last convolutional layer
    """
    def __init__(self, params_loc=None):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

        if params_loc:
            self.model.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        last_block = self.model.layer4[-1]  # Last BasicBlock of layer 3
        return last_block.conv2


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
        x = relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def get_last_conv_layer(self):
        return self.conv2

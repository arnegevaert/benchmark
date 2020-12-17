from torchvision import datasets, transforms
import os
from os import path
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from experiments.medical_imaging.lib.datasets import Aptos, Cxr8, CBIS_DDSM_patches, HAM10000,PcamDataset, MedMNIST
from experiments.medical_imaging.lib.models import Alexnet, Densenet, Resnet, EfficientNet, MedMnist_ResNet18

_DATA_LOC = os.environ["BM_DATA_LOC"] if "BM_DATA_LOC" in os.environ else path.join(path.dirname(__file__), "../../data")



def get_dataset_model(name):
    if name == "Aptos":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = Aptos(path.join(_DATA_LOC, "APTOS"), train=False,img_size=320)
        model = Densenet("densenet121", 5, path.join(_DATA_LOC, "models/Aptos/densenet121.pt"))
        sample_shape = (320,320)
    elif name == "CXR8":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4821, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        ds = Cxr8(path.join(_DATA_LOC, "CXR8"), train=False)
        model = Densenet("densenet121", 14, path.join(_DATA_LOC, "models/CXR8/densenet121.pt"))
        sample_shape = (512, 512)
    elif name == "CBIS-DDSM":
        ds = CBIS_DDSM_patches(path.join(_DATA_LOC,"CBIS-DDSM"), train=False,imsize=224)
        model = Resnet("resnet50", 2, path.join(_DATA_LOC, "models/CBIS-DDSM/resnet50.pt"))
        sample_shape = (224, 224)
    elif name == "HAM10000":
        ds = HAM10000(path.join(_DATA_LOC, "HAM10000"),train=False,)
        model = EfficientNet('efficientnet-b0',7,params_loc=path.join(_DATA_LOC, "models/HAM10000/efficientnet-b0.pt"))
        sample_shape = (224,224)
    elif name == "PCAM":
        ds = PcamDataset(path.join(_DATA_LOC, "PCAM"),train=False)
        model = Densenet('densenet121',2, params_loc=path.join(_DATA_LOC, "models/PCAM/densenet121.pt"))
        sample_shape = (96,96)
    elif name == "PathMNIST":
        ds = MedMNIST(path.join(_DATA_LOC, "MedMNIST"),"pathmnist",split='test')
        model = MedMnist_ResNet18(num_classes=9, params_loc=path.join(_DATA_LOC, "models/MedMNIST/pathmnist.pt"))
        sample_shape=(28,28)
    elif name == "OCTMNIST":
        ds = MedMNIST(path.join(_DATA_LOC, "MedMNIST"), "octmnist", split='test')
        model = MedMnist_ResNet18(num_classes=4, params_loc=path.join(_DATA_LOC, "models/MedMNIST/octmnist.pt"), in_channels=1)
        sample_shape = (28, 28)
    elif name == "ChestMNIST":
        ds = MedMNIST(path.join(_DATA_LOC, "MedMNIST"), "chestmnist", split='test')
        model = MedMnist_ResNet18(num_classes=14, params_loc=path.join(_DATA_LOC, "models/MedMNIST/chestmnist.pt"), in_channels=1)
        sample_shape = (28, 28)
    else:
        raise ValueError(f"Invalid dataset: {name}")
    return ds, model, sample_shape

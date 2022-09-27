from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch
from .imagenet_subset import ImagenetSubset
from os import path


def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ImagenetSubset(path.join("data", "imagenette2", "val"), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model

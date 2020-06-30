import torch
from attrbench.models import Densenet, Resnet, Alexnet, Vgg, Mobilenet_v2, Squeezenet, BasicCNN
from attrbench.datasets import *
from os import path


data_root = "../data"
models_root = path.join(data_root, "models")

all_models = {
    "Aptos": {
        "densenet121": Densenet(version="densenet121", output_logits=True, num_classes=5,
                                 params_loc=path.join(models_root, "Aptos", "densenet121.pt"))
    },
    "ImageNette": {
        "alexnet": Alexnet(output_logits=True, num_classes=10,
                           params_loc=path.join(models_root, "ImageNette", "alexnet.pt")),
        "densenet": Densenet(version="densenet121", output_logits=True, num_classes=10,
                             params_loc=path.join(models_root, "ImageNette", "densenet.pt")),
        "mobilenet_v2": Mobilenet_v2(output_logits=True, num_classes=10,
                                     params_loc=path.join(models_root, "ImageNette", "mobilenet_v2.pt")),
        "resnet18": Resnet(version="resnet18", num_classes=10, output_logits=True,
                           params_loc=path.join(models_root, "ImageNette", "resnet18.pt")),
        "squeezenet1_0": Squeezenet(version="squeezenet1_0", output_logits=True, num_classes=10,
                                    params_loc=path.join(models_root, "ImageNette", "squeezenet1_0.pt")),
        "vgg11_bn": Vgg(version="vgg11_bn", output_logits=True, num_classes=10,
                        params_loc=path.join(models_root, "ImageNette", "vgg11_bn.pt"))
    },
    "CIFAR10": {
        
    },
    "MNIST": {"cnn": BasicCNN(output_logits=True, num_classes=10,
                              params_loc=path.join(models_root, "MNIST", "cnn.pt"))}
}

print("All models OK")

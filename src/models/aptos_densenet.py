"""
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
In CVPR, 2016.
Based on: https://github.com/chenyaofo/CIFAR-pretrained-models/blob/master/cifar_pretrainedmodels/resnet.py
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models import ConvolutionalNetworkModel
from os import path

model_constructor_dict = {"densenet121": torchvision.models.densenet121}


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        base_model = model_constructor_dict[model](pretrained=True, progress=True)
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Linear(1024, 5),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Global avg pooling
        x = torch.flatten(x, 1)
        return self.classifier(x)


pretrained_settings = {
    "aptos": {
        'densenet121': 'aptos_densenet121_weights.pt'
    }
}


class AptosDensenet(ConvolutionalNetworkModel):
    def __init__(self, dataset="aptos", densenet="densenet121", device='cuda'):
        super().__init__()
        if dataset not in ["aptos"]:
            raise ValueError("dataset must be in {aptos}")
        if densenet not in ["densenet121"]:
            raise ValueError("densenet must be in {densenet121}")
        params_loc = path.join(path.dirname(__file__), "saved_models", pretrained_settings[dataset][densenet])

        self.net = Net(densenet).to(device)
        if not path.exists(params_loc):
            raise FileNotFoundError(f"{params_loc} does not exist. "
                                    f"Use the train_aptos_densenet.py script to train and save weights.")
        self.net.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def get_last_conv_layer(self) -> nn.Module:
        last_block = self.net.features.transition3  # Last BasicBlock of layer 3
        return last_block.conv  # Second convolutional layer of last BasicBlock

    def get_conv_net(self) -> nn.Module:
        return self.net

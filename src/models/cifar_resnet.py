"""
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
Deep Residual Learning for Image Recognition.
In CVPR, 2016.
Based on: https://github.com/chenyaofo/CIFAR-pretrained-models/blob/master/cifar_pretrainedmodels/resnet.py
"""

import torch
import urllib
import torch.nn as nn
from models import ConvolutionalNetworkModel
from os import path


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Net, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.softmax(self.fc(x))


base_url = 'https://github.com/chenyaofo/CIFAR-pretrained-models/releases/download/resnet/'
pretrained_settings = {
    "cifar10": {
        'resnet20': 'cifar10-resnet20-30abc31d.pth', 'resnet32': 'cifar10-resnet32-e96f90cf.pth',
        'resnet44': 'cifar10-resnet44-f2c66da5.pth', 'resnet56': 'cifar10-resnet56-f5939a66.pth',
        'num_classes': 10
    },
    "cifar100": {
        'resnet20': 'cifar100-resnet20-8412cc70.pth', 'resnet32': 'cifar100-resnet32-6568a0a0.pth',
        'resnet44': 'cifar100-resnet44-20aaa8cf.pth', 'resnet56': 'cifar100-resnet56-2f147f26.pth',
        'num_classes': 100
    }
}


class CifarResNet(ConvolutionalNetworkModel):
    def __init__(self, dataset="cifar10", resnet="resnet20"):
        super().__init__()
        if dataset not in ["cifar10", "cifar100"]:
            raise ValueError("dataset must be in {cifar10, cifar100}")
        if resnet not in ["resnet20", "resnet32", "resnet44", "resnet56"]:
            raise ValueError("resnet must be in {resnet20, resnet32, resnet44, resnet56}")
        params_loc = path.join(path.dirname(__file__), "saved_models", pretrained_settings[dataset][resnet])
        layers = {
            "resnet20": [3, 3, 3], "resnet32": [5, 5, 5],
            "resnet44": [7, 7, 7], "resnet56": [9, 9, 9]
        }
        self.net = Net(BasicBlock, layers[resnet], num_classes=10 if dataset is "cifar10" else 100)
        if not path.exists(params_loc):
            url = base_url + pretrained_settings[dataset][resnet]
            print(f"Downloading parameters from {url}...")
            urllib.request.urlretrieve(url, params_loc)
            print(f"Download finished.")
        self.net.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def get_last_conv_layer(self) -> nn.Module:
        last_block = self.net.layer3[-1]  # Last BasicBlock of layer 3
        return last_block.conv2  # Second convolutional layer of last BasicBlock

    def get_conv_net(self) -> nn.Module:
        return self.net

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)



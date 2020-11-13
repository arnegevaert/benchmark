import torch
import torchvision
import torch.nn as nn

versions = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
            'resnet152']


class Resnet(nn.Module):
    def __init__(self, version, num_classes, params_loc=None,pretrained=False):
        super().__init__()
        if version not in versions:
            raise NotImplementedError('version not supported')
        self.version = version
        fn = getattr(torchvision.models, version)
        model = fn(pretrained=pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(model.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        last_block = self.layer4[-1]  # Last BasicBlock of layer 3
        return last_block.conv2

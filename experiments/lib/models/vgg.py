import torch
import torchvision
import torch.nn as nn

versions = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19'] # all vgg models from torchvision model zoo can be used


class Vgg(nn.Module):
    def __init__(self, version, num_classes, params_loc=None):
        super().__init__()
        if version not in versions:
            raise NotImplementedError('version not supported')
        self.version = version
        fn = getattr(torchvision.models, version)
        model = fn(pretrained=False)
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        if self.version[-2:] == "bn":
            return self.features[-4]
        else:
            return self.features[-3]

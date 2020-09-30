import torch
import torchvision
import torch.nn as nn


class Alexnet(nn.Module):
    def __init__(self, num_classes, params_loc=None):
        super().__init__()
        model = torchvision.models.alexnet()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.classifier[-1] = nn.Linear(4096, num_classes)
        self.softmax = nn.Softmax(dim=1)
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
        return self.features[-3]
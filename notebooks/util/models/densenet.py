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


class Densenet(nn.Module):
    def __init__(self, output_logits, num_classes ,params_loc=None):
        super(Densenet, self).__init__()
        base_model = torchvision.models.densenet121(pretrained=False, progress=True)
        self.features = base_model.features
        self.classifier = base_model.classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes))
        self.output_logits = output_logits
        self.softmax = nn.Softmax(dim=1)

        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def get_logits(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global avg pooling
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        if self.output_logits:
            return logits
        return self.softmax(logits)

    def get_last_conv_layer(self) -> nn.Module:
        last_block = self.features.transition3  # Last BasicBlock of layer 3
        return last_block.conv  # Second convolutional layer of last BasicBlock

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.base_model.to(*args, **kwargs)


def AptosDensenet(output_logits, params_loc=None):
    return Densenet(output_logits, 5, params_loc)


def ImagenetteDensenet(output_logits, params_loc=None):
    return Densenet(output_logits, 10, params_loc)
from efficientnet_pytorch import EfficientNet as _EfficientNet
import torch
import torch.nn as nn


versions = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4',
    'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'efficientnet-b8',]


class EfficientNet(_EfficientNet):
    def __init__(self,version, num_classes, params_loc=None):

        self.__dict__ = _EfficientNet.from_pretrained(version, num_classes=num_classes).__dict__ #dirty way to wrap EfficientNet


        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def get_last_conv_layer(self) -> nn.Module:
        return self._conv_head


if __name__ == '__main__':
    #debugging
    model = EfficientNet("efficientnet-b0",num_classes=7)
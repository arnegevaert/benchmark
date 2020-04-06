import torch.nn as nn


class Model:
    # Expects input to have batch dimension
    def predict(self, x):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        raise NotImplementedError


class ConvolutionalNetworkModel(Model):
    def predict(self, x):
        raise NotImplementedError

    def get_last_conv_layer(self) -> nn.Module:
        raise NotImplementedError

    def get_conv_net(self) -> nn.Module:
        raise NotImplementedError

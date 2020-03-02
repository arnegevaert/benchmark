from methods import Method
from captum import attr
import torch.nn as nn
import torch


class Gradient(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.saliency = attr.Saliency(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.saliency.attribute(x, target=target)


class InputXGradient(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.input_x_grad = attr.InputXGradient(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.input_x_grad.attribute(x, target=target)


class IntegratedGradients(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.integrated_gradients = attr.IntegratedGradients(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.integrated_gradients.attribute(x, target=target)


class DeepLift(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.deeplift = attr.DeepLift(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.deeplift.attribute(x, target=target)


# This is not really an attribution technique, just to establish a baseline
class Random(Method):
    def __init__(self):
        super().__init__()

    def attribute(self, x, target):
        return torch.rand_like(x)

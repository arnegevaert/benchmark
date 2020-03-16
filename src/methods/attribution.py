from methods import Method
from captum import attr
import torch.nn as nn
import torch

# TODO lot of duplicated code here

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


class GuidedBackprop(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.guided_backprop = attr.GuidedBackprop(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.guided_backprop.attribute(x, target=target)


class Deconvolution(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.deconvolution = attr.Deconvolution(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.deconvolution.attribute(x, target=target)


class GuidedGradCAM(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.guided_gradcam = attr.GuidedGradCam(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.guided_gradcam.attribute(x, target=target)


# TODO by default, DeepLift is equivalent to InputXGradient. Read DeepLift paper for more details.
"""
class DeepLift(Method):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.deeplift = attr.DeepLift(net)

    def attribute(self, x, target):
        self.net.eval()
        return self.deeplift.attribute(x, target=target)
"""

# This is not really an attribution technique, just to establish a baseline
class Random(Method):
    def __init__(self):
        super().__init__()

    def attribute(self, x, target):
        return torch.rand_like(x)

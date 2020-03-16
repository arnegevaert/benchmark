from methods import Method
from models import ConvolutionalNetworkModel
from captum import attr
import torch.nn as nn
import torch


class SimpleCaptumMethod(Method):
    METHODS = {
        "Gradient": attr.Saliency,
        "InputXGradient": attr.InputXGradient,
        "IntegratedGradients": attr.IntegratedGradients,
        "GuidedBackprop": attr.GuidedBackprop,
        "Deconvolution": attr.Deconvolution,
        "Ablation": attr.FeatureAblation,
    }

    def __init__(self, net: nn.Module, method: str):
        super().__init__()
        self.net = net
        self.method = SimpleCaptumMethod.METHODS[method](net)

    def attribute(self, x, target):
        self.net.eval()
        return self.method.attribute(x, target=target)


class GuidedGradCAM(Method):
    def __init__(self, model: ConvolutionalNetworkModel):
        super().__init__()
        self.net = model.get_conv_net()
        self.guided_gradcam = attr.GuidedGradCam(self.net, model.get_last_conv_layer())

    def attribute(self, x, target):
        self.net.eval()
        return self.guided_gradcam.attribute(x, target=target)


class Occlusion(Method):
    def __init__(self, net: nn.Module, sliding_window_shapes):
        super().__init__()
        self.net = net
        self.occlusion = attr.Occlusion(net)
        self.sliding_window_shapes = sliding_window_shapes

    def attribute(self, x, target):
        self.net.eval()
        return self.occlusion.attribute(x, target=target, sliding_window_shapes=self.sliding_window_shapes)


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

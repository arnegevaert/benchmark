from methods import Method
from models import ConvolutionalNetworkModel
from captum import attr
import torch.nn as nn
import torch
import warnings


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
        self.name = method

    def attribute(self, x, target):
        batch_size = x.shape[0]
        sample_shape = x.shape[1:]
        self.net.eval()
        attrs = self.method.attribute(x, target=target)
        flattened_attrs = attrs.reshape(batch_size, -1)
        min_per_img = flattened_attrs.min(dim=-1)[0].unsqueeze(dim=-1)
        max_per_img = flattened_attrs.max(dim=-1)[0].unsqueeze(dim=-1)
        normalized = (flattened_attrs - min_per_img) / (max_per_img - min_per_img)  # Normalize: [0,1] per image
        if torch.any(torch.isnan(normalized)):
            warnings.warn(f"NaNs detected in {self.name} attributions: replaced by 0.")
            normalized[torch.where(torch.isnan(normalized))] = 0
        result = normalized.reshape((batch_size, *sample_shape))
        return result


class GuidedGradCAM(Method):
    def __init__(self, model: ConvolutionalNetworkModel):
        super().__init__()
        self.net = model.get_conv_net()
        self.guided_gradcam = attr.GuidedGradCam(self.net, model.get_last_conv_layer())

    def attribute(self, x, target):
        batch_size = x.shape[0]
        sample_shape = x.shape[1:]
        self.net.eval()
        attrs = self.guided_gradcam.attribute(x, target=target)
        flattened_attrs = attrs.reshape(batch_size, -1)
        min_per_img = flattened_attrs.min(dim=-1)[0].unsqueeze(dim=-1)
        max_per_img = flattened_attrs.max(dim=-1)[0].unsqueeze(dim=-1)
        normalized = (flattened_attrs - min_per_img) / (max_per_img - min_per_img + 1e-20)  # Normalize: [0,1] per image
        result = normalized.reshape((batch_size, *sample_shape))
        return result


class Occlusion(Method):
    def __init__(self, net: nn.Module, sliding_window_shapes):
        super().__init__()
        self.net = net
        self.occlusion = attr.Occlusion(net)
        self.sliding_window_shapes = sliding_window_shapes

    def attribute(self, x, target):
        self.net.eval()
        attrs = self.occlusion.attribute(x, target=target, sliding_window_shapes=self.sliding_window_shapes)
        denom = attrs.max() - attrs.min()
        if denom == 0:
            warnings.warn("Occlusion: denominator is 0. Values will be nan.")
        return (attrs - attrs.min()) / (attrs.max() - attrs.min())  # Normalize: [0,1]


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

from methods import Method
from models import ConvolutionalNetworkModel
from captum import attr
import torch.nn as nn
import torch
import warnings


def _normalize_attributions(attrs):
    abs_attrs = attrs.reshape(attrs.shape[0], -1).abs()
    max_abs_attr_per_image = abs_attrs.max(dim=-1)[0]
    if torch.any(max_abs_attr_per_image == 0):
        warnings.warn("Completely 0 attributions returned for sample.")
        # If an image has 0 max abs attr, all attrs are 0 for that image
        # Divide by 1 to return the original constant 0 attributions
        max_abs_attr_per_image[torch.where(max_abs_attr_per_image == 0)] = 1
    # Add as many singleton dimensions to max_abs_attr_per_image as necessary to divide
    while len(max_abs_attr_per_image.shape) < len(attrs.shape):
        max_abs_attr_per_image = max_abs_attr_per_image.unsqueeze(-1)
    normalized = attrs / max_abs_attr_per_image
    return normalized.reshape(attrs.shape)


class SimpleCaptumMethod(Method):
    METHODS = {
        # TODO set the values for abs according to the actual method properties
        "Gradient": {"constructor": attr.Saliency, "abs": True},  # Gradient abs can be set to False by passing abs=False in constructor kwargs
        "InputXGradient": {"constructor": attr.InputXGradient, "abs": False},
        "IntegratedGradients": {"constructor": attr.IntegratedGradients, "abs": False},
        "GuidedBackprop": {"constructor": attr.GuidedBackprop, "abs": False},
        "Deconvolution": {"constructor": attr.Deconvolution, "abs": False},
        "Ablation": {"constructor": attr.FeatureAblation, "abs": False}
    }
    # optional arguments
    METHOD_KWAGS = {
        "Gradient": {},
        "InputXGradient": {},
        "IntegratedGradients": {"internal_batch_size": 8},# would be better to 
        "GuidedBackprop": {},
        "Deconvolution": {},
        "Ablation": {},

    }

    def __init__(self, net: nn.Module, method: str, normalize=True, **kwargs):
        super().__init__()
        self.net = net
        self.name = method
        method_dict = SimpleCaptumMethod.METHODS[self.name]
        self.method = method_dict["constructor"](net, **kwargs)
        self.is_absolute = method_dict["abs"] if "abs" not in kwargs else kwargs["abs"]
        self.normalize = normalize
        self.method_kwargs = SimpleCaptumMethod.METHOD_KWAGS[self.name]


    def attribute(self, x, target):
        self.net.eval()
        attrs = self.method.attribute(x, target=target, **self.method_kwargs)
        if self.normalize:
            return _normalize_attributions(attrs)
        return attrs


class GuidedGradCAM(Method):
    def __init__(self, model: ConvolutionalNetworkModel, normalize=True):
        super().__init__()
        self.net = model.get_conv_net()
        self.guided_gradcam = attr.GuidedGradCam(self.net, model.get_last_conv_layer())
        self.normalize = normalize
        self.is_absolute = False

    def attribute(self, x, target):
        self.net.eval()
        attrs = self.guided_gradcam.attribute(x, target=target)
        if self.normalize:
            return _normalize_attributions(attrs)
        return attrs


class Occlusion(Method):
    def __init__(self, net: nn.Module, sliding_window_shapes, normalize=True):
        super().__init__()
        self.net = net
        self.occlusion = attr.Occlusion(net)
        self.sliding_window_shapes = sliding_window_shapes
        self.normalize = normalize
        self.is_absolute = False

    def attribute(self, x, target):
        self.net.eval()
        attrs = self.occlusion.attribute(x, target=target, sliding_window_shapes=self.sliding_window_shapes)
        if self.normalize:
            return _normalize_attributions(attrs)
        return attrs


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
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.is_absolute = False

    def attribute(self, x, target):
        attrs = torch.rand_like(x) * 2 - 1
        if self.normalize:
            return _normalize_attributions(attrs)
        return attrs

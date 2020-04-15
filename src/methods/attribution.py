from methods import Method
from models import ConvolutionalNetworkModel
from captum import attr
import torch.nn as nn
import torch
import warnings
import numpy as np


def _normalize_attributions(attrs):
    abs_attrs = np.abs(attrs.reshape(attrs.shape[0], -1))
    max_abs_attr_per_image = np.max(abs_attrs, axis=1)[0]
    if np.any(max_abs_attr_per_image == 0):
        warnings.warn("Completely 0 attributions returned for sample.")
        # If an image has 0 max abs attr, all attrs are 0 for that image
        # Divide by 1 to return the original constant 0 attributions
        max_abs_attr_per_image[np.where(max_abs_attr_per_image == 0)] = 1
    # Add as many singleton dimensions to max_abs_attr_per_image as necessary to divide
    while len(max_abs_attr_per_image.shape) < len(attrs.shape):
        max_abs_attr_per_image = np.expand_dims(max_abs_attr_per_image, axis=-1)
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

    def __init__(self, net: nn.Module, method: str, normalize=True, **kwargs):
        super().__init__()
        self.net = net
        self.name = method
        method_dict = SimpleCaptumMethod.METHODS[self.name]
        self.method = method_dict["constructor"](net, **kwargs)
        self.is_absolute = method_dict["abs"] if "abs" not in kwargs else kwargs["abs"]
        self.normalize = normalize

    def __call__(self, x: np.ndarray, target: np.ndarray):
        self.net.eval()
        attrs = self.method.attribute(torch.tensor(x), target=torch.tensor(target)).detach().numpy()
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

    def __call__(self, x: np.ndarray, target: np.ndarray):
        self.net.eval()
        attrs = self.guided_gradcam.attribute(torch.tensor(x), target=torch.tensor(target)).detach().numpy()
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

    def __call__(self, x: np.ndarray, target: np.ndarray):
        self.net.eval()
        attrs = self.occlusion.attribute(torch.tensor(x), target=torch.tensor(target),
                                         sliding_window_shapes=self.sliding_window_shapes).detach().numpy()
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

    def __call__(self, x: np.ndarray, target: np.ndarray):
        attrs = np.random.rand(*x.shape) * 2 - 1
        if self.normalize:
            return _normalize_attributions(attrs)
        return attrs

from captum import attr
import torch.nn as nn
import torch
from .util import normalize_attributions


class CaptumMethod:
    def __init__(self, method: attr.Attribution, normalize, absolute=False):
        self.normalize = normalize
        self.method = method
        self.is_absolute = absolute

    def _attribute(self, x, target):
        return self.method.attribute(x, target=target)

    def __call__(self, x, target):
        attrs = self._attribute(x, target)
        if self.normalize:
            return normalize_attributions(attrs)
        return attrs


class Gradient(CaptumMethod):
    def __init__(self, forward_func, normalize=True):
        super(Gradient, self).__init__(attr.Saliency(forward_func), normalize, absolute=True)


class InputXGradient(CaptumMethod):
    def __init__(self, forward_func, normalize=True):
        super(InputXGradient, self).__init__(attr.InputXGradient(forward_func), normalize)


class IntegratedGradients(CaptumMethod):
    def __init__(self, forward_func, normalize=True):
        super(IntegratedGradients, self).__init__(attr.IntegratedGradients(forward_func), normalize)


class GuidedBackprop(CaptumMethod):
    def __init__(self, model: nn.Module, normalize=True):
        super(GuidedBackprop, self).__init__(attr.GuidedBackprop(model), normalize)


class Deconvolution(CaptumMethod):
    def __init__(self, model: nn.Module, normalize=True):
        super(Deconvolution, self).__init__(attr.Deconvolution(model), normalize)


class Ablation(CaptumMethod):
    def __init__(self, forward_func, normalize=True, baselines=None, feature_mask=None, perturbations_per_eval=1):
        super(Ablation, self).__init__(attr.FeatureAblation(forward_func), normalize)
        self.baselines = baselines
        self.feature_mask = feature_mask
        self.perturbations_per_eval = perturbations_per_eval

    def _attribute(self, x, target):
        return self.method.attribute(x, target=target, baselines=self.baselines, feature_mask=self.feature_mask,
                                     perturbations_per_eval=self.perturbations_per_eval)


class Occlusion(CaptumMethod):
    def __init__(self, forward_func, sliding_window_shapes, normalize=True):
        super(Occlusion, self).__init__(attr.Occlusion(forward_func), normalize)
        self.sliding_window_shapes = sliding_window_shapes

    def _attribute(self, x, target):
        return self.method.attribute(x, target=target, sliding_window_shapes=self.sliding_window_shapes)


class GuidedGradCAM(CaptumMethod):
    def __init__(self, model: nn.Module, layer: nn.Module, normalize=True):
        super(GuidedGradCAM, self).__init__(attr.GuidedGradCam(model, layer), normalize)


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
class Random:
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.is_absolute = False

    def __call__(self, x, target):
        attrs = torch.rand(*x.shape) * 2 - 1
        if self.normalize:
            return normalize_attributions(attrs)
        return attrs

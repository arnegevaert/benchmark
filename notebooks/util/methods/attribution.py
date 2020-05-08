from captum import attr
import torch.nn as nn
import torch
from .util import normalize_attributions
from skimage.filters import sobel


class AttributionMethod:
    def __init__(self, method: attr.Attribution, normalize, absolute=False, aggregation_fn=None):
        self.normalize = normalize
        self.method = method
        self.is_absolute = absolute
        aggregation_fns = {
            "max": lambda x: torch.max(x, dim=1),
            "avg": lambda x: torch.mean(x, dim=1)
        }
        self.aggregation_fn = aggregation_fns.get(aggregation_fn, None)

    def _attribute(self, x, target):
        return self.method.attribute(x, target=target)

    def __call__(self, x, target):
        attrs = self._attribute(x, target)
        if self.aggregation_fn:
            attrs = self.aggregation_fn(attrs)
        if self.normalize:
            attrs = normalize_attributions(attrs)
        return attrs


class Gradient(AttributionMethod):
    def __init__(self, forward_func, normalize=True, aggregation_fn=None):
        super(Gradient, self).__init__(attr.Saliency(forward_func), normalize, absolute=True,
                                       aggregation_fn=aggregation_fn)


class InputXGradient(AttributionMethod):
    def __init__(self, forward_func, normalize=True, aggregation_fn=None):
        super(InputXGradient, self).__init__(attr.InputXGradient(forward_func), normalize,
                                             aggregation_fn=aggregation_fn)


class IntegratedGradients(AttributionMethod):
    def __init__(self, forward_func, normalize=True, aggregation_fn=None):
        super(IntegratedGradients, self).__init__(attr.IntegratedGradients(forward_func), normalize,
                                                  aggregation_fn=aggregation_fn)


class GuidedBackprop(AttributionMethod):
    def __init__(self, model: nn.Module, normalize=True, aggregation_fn=None):
        super(GuidedBackprop, self).__init__(attr.GuidedBackprop(model), normalize, aggregation_fn=aggregation_fn)


class Deconvolution(AttributionMethod):
    def __init__(self, model: nn.Module, normalize=True, aggregation_fn=None):
        super(Deconvolution, self).__init__(attr.Deconvolution(model), normalize, aggregation_fn=aggregation_fn)


class Ablation(AttributionMethod):
    def __init__(self, forward_func, normalize=True, baselines=None, feature_mask=None, perturbations_per_eval=1,
                 aggregation_fn=None):
        super(Ablation, self).__init__(attr.FeatureAblation(forward_func), normalize, aggregation_fn=None)
        self.baselines = baselines
        self.feature_mask = feature_mask
        self.perturbations_per_eval = perturbations_per_eval

    def _attribute(self, x, target):
        return self.method.attribute(x, target=target, baselines=self.baselines, feature_mask=self.feature_mask,
                                     perturbations_per_eval=self.perturbations_per_eval)


class Occlusion(AttributionMethod):
    def __init__(self, forward_func, sliding_window_shapes, normalize=True, aggregation_fn=None):
        super(Occlusion, self).__init__(attr.Occlusion(forward_func), normalize, aggregation_fn=None)
        self.sliding_window_shapes = sliding_window_shapes

    def _attribute(self, x, target):
        return self.method.attribute(x, target=target, sliding_window_shapes=self.sliding_window_shapes)


class GuidedGradCAM(AttributionMethod):
    def __init__(self, model: nn.Module, layer: nn.Module, normalize=True, aggregation_fn=None):
        super(GuidedGradCAM, self).__init__(attr.GuidedGradCam(model, layer), normalize, aggregation_fn=None)


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
    def __init__(self, normalize=True, aggregation_fn=None):
        super().__init__()
        self.normalize = normalize
        self.is_absolute = False
        aggregation_fns = {
            "max": lambda x: torch.max(x, dim=1),
            "avg": lambda x: torch.mean(x, dim=1)
        }
        self.aggregation_fn = aggregation_fns.get(aggregation_fn, None)

    def __call__(self, x, target):
        attrs = torch.rand(*x.shape) * 2 - 1
        if self.aggregation_fn:
            attrs = self.aggregation_fn(attrs)
        if self.normalize:
            return normalize_attributions(attrs)
        return attrs


class EdgeDetection:
    def __init__(self, normalize=True, aggregation_fn=None):
        super().__init__()
        self.normalize = normalize
        self.is_absolute = False
        aggregation_fns = {
            "max": lambda x: torch.max(x, dim=1),
            "avg": lambda x: torch.mean(x, dim=1)
        }
        self.aggregation_fn = aggregation_fns.get(aggregation_fn, None)

    def __call__(self, x, target):
        device = x.device
        x = x.detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        for i in range(x.shape[0]):
            for channel in range(x.shape[1]):
                x[i, channel] = sobel(x[i, channel])
        attrs = torch.tensor(x).to(device)

        if self.aggregation_fn:
            attrs = self.aggregation_fn(attrs)
        if self.normalize:
            return normalize_attributions(attrs)
        return attrs

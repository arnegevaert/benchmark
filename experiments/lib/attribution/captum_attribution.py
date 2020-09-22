from .attribution import AttributionMethod
from captum import attr
import torch.nn as nn


class CaptumMethod(AttributionMethod):
    def __init__(self, method: attr.Attribution, absolute, normalize=True, aggregation_fn=None, **kwargs):
        super(CaptumMethod, self).__init__(absolute, normalize, aggregation_fn)
        self.method = method
        self.attribute_kwargs = kwargs

    def _attribute(self, x, target, **kwargs):
        return self.method.attribute(x, target=target, **self.attribute_kwargs)


class Gradient(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        super(Gradient, self).__init__(attr.Saliency(forward_func), True, **kwargs)


class SmoothGrad(CaptumMethod):
    # useful kwargs: n_samples, abs=False
    def __init__(self, forward_func, **kwargs):
        method = attr.NoiseTunnel(attr.Saliency(forward_func))
        super(SmoothGrad, self).__init__(method, True, **kwargs)


class InputXGradient(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        super(InputXGradient, self).__init__(attr.InputXGradient(forward_func), False, **kwargs)


class IntegratedGradients(CaptumMethod):
    # useful kwargs: n_steps
    def __init__(self, forward_func, **kwargs):
        super(IntegratedGradients, self).__init__(attr.IntegratedGradients(forward_func), False, **kwargs)


class SmoothIntegratedGradients(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        method = attr.NoiseTunnel(attr.IntegratedGradients(forward_func))
        super(SmoothIntegratedGradients, self).__init__(method, False, **kwargs)


class GuidedBackprop(CaptumMethod):
    def __init__(self, model: nn.Module, **kwargs):
        super(GuidedBackprop, self).__init__(attr.GuidedBackprop(model), False, **kwargs)


class Deconvolution(CaptumMethod):
    def __init__(self, model: nn.Module, **kwargs):
        super(Deconvolution, self).__init__(attr.Deconvolution(model), False, **kwargs)


class Ablation(CaptumMethod):
    def __init__(self, forward_func, baselines=None, feature_mask=None, perturbations_per_eval=1, **kwargs):
        super(Ablation, self).__init__(attr.FeatureAblation(forward_func), False, **kwargs)
        self.baselines = baselines
        self.feature_mask = feature_mask
        self.perturbations_per_eval = perturbations_per_eval

    def _attribute(self, x, target, **kwargs):
        return self.method.attribute(x, target=target, baselines=self.baselines, feature_mask=self.feature_mask,
                                     perturbations_per_eval=self.perturbations_per_eval)


class Occlusion(CaptumMethod):
    def __init__(self, forward_func, sliding_window_shapes, **kwargs):
        super(Occlusion, self).__init__(attr.Occlusion(forward_func), False, **kwargs)
        self.sliding_window_shapes = sliding_window_shapes

    def _attribute(self, x, target, **kwargs):
        return self.method.attribute(x, target=target, sliding_window_shapes=self.sliding_window_shapes)


class GuidedGradCAM(AttributionMethod):
    """
    GuidedGradCAM is just element-wise product of guided backprop and gradCAM.
    Captum implementation multiplies with only non-negative elements of gradCAM which can result in constant
    zero saliency maps.
    """
    def __init__(self, model: nn.Module, layer: nn.Module, upsample_shape, **kwargs):
        super().__init__(False, **kwargs)
        self.model = model
        self.layer = layer
        self.gbp = attr.GuidedBackprop(model)
        self.gcam = GradCAM(model, layer, upsample_shape)

    def _attribute(self, x, target, **kwargs):
        gcam_attrs = self.gcam(x, target)
        gbp_attrs = self.gbp.attribute(x, target)
        return gcam_attrs * gbp_attrs


class GradCAM(CaptumMethod):
    def __init__(self, model: nn.Module, layer: nn.Module, upsample_shape, **kwargs):
        self.upsample_shape = upsample_shape
        super().__init__(attr.LayerGradCam(model, layer), False, **kwargs)

    def _attribute(self, x, target, **kwargs):
        attrs = self.method.attribute(x, target, relu_attributions=False)
        # Upsample attributions
        return attr.LayerAttribution.interpolate(attrs, self.upsample_shape, interpolate_mode="bilinear")


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

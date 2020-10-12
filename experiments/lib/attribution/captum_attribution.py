from experiments.lib.attribution import AttributionMethod
from captum import attr
import torch.nn as nn


class CaptumMethod(AttributionMethod):
    def __init__(self, method: attr.Attribution, normalize=False, aggregation_fn=None, **kwargs):
        super(CaptumMethod, self).__init__(normalize, aggregation_fn)
        self.method = method
        self.attribute_kwargs = kwargs

    def _attribute(self, x, target, **kwargs):
        return self.method.attribute(x, target=target, **self.attribute_kwargs)


class Gradient(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        super(Gradient, self).__init__(attr.Saliency(forward_func), **kwargs)


class SmoothGrad(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        method = attr.NoiseTunnel(attr.Saliency(forward_func))
        super(SmoothGrad, self).__init__(method, **kwargs)


class InputXGradient(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        super(InputXGradient, self).__init__(attr.InputXGradient(forward_func), **kwargs)


class IntegratedGradients(CaptumMethod):
    def __init__(self, forward_func, internal_batch_size=None, **kwargs):
        super(IntegratedGradients, self).__init__(attr.IntegratedGradients(forward_func), **kwargs)
        self.internal_batch_size = internal_batch_size

    def _attribute(self, x, target, **kwargs):
        return self.method.attribute(x, target=target, internal_batch_size=self.internal_batch_size)


class SmoothIntegratedGradients(CaptumMethod):
    def __init__(self, forward_func, **kwargs):
        method = attr.NoiseTunnel(attr.IntegratedGradients(forward_func))
        super(SmoothIntegratedGradients, self).__init__(method, **kwargs)


class GuidedBackprop(CaptumMethod):
    def __init__(self, model: nn.Module, **kwargs):
        super(GuidedBackprop, self).__init__(attr.GuidedBackprop(model), **kwargs)


class Deconvolution(CaptumMethod):
    def __init__(self, model: nn.Module, **kwargs):
        super(Deconvolution, self).__init__(attr.Deconvolution(model), **kwargs)


class Ablation(CaptumMethod):
    def __init__(self, forward_func, baselines=None, feature_mask=None, perturbations_per_eval=1, **kwargs):
        super(Ablation, self).__init__(attr.FeatureAblation(forward_func), **kwargs)
        self.baselines = baselines
        self.feature_mask = feature_mask
        self.perturbations_per_eval = perturbations_per_eval

    def _attribute(self, x, target, **kwargs):
        return self.method.attribute(x, target=target, baselines=self.baselines, feature_mask=self.feature_mask,
                                     perturbations_per_eval=self.perturbations_per_eval)


class Occlusion(CaptumMethod):
    def __init__(self, forward_func, sliding_window_shapes, **kwargs):
        super(Occlusion, self).__init__(attr.Occlusion(forward_func), **kwargs)
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
        super().__init__(**kwargs)
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
        super().__init__(attr.LayerGradCam(model, layer), **kwargs)

    def _attribute(self, x, target, **kwargs):
        attrs = self.method.attribute(x, target, relu_attributions=False)
        # Upsample attributions
        return attr.LayerAttribution.interpolate(attrs, self.upsample_shape, interpolate_mode="bilinear")

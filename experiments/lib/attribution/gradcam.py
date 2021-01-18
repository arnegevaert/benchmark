from captum import attr
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module, upsample_shape):
        self.upsample_shape = upsample_shape
        self.method = attr.LayerGradCam(model, layer)

    def __call__(self, x, target):
        attrs = self.method.attribute(x, target, relu_attributions=True)
        # Upsample attributions
        return attr.LayerAttribution.interpolate(attrs, self.upsample_shape, interpolate_mode="bilinear")
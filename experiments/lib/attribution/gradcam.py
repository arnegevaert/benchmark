from experiments.lib.attribution import CaptumMethod
from captum import attr
from torch import nn


class GradCAM(CaptumMethod):
    def __init__(self, model: nn.Module, layer: nn.Module, upsample_shape):
        self.upsample_shape = upsample_shape
        super().__init__(attr.LayerGradCam(model, layer))

    def _attribute(self, x, target):
        attrs = self.method.attribute(x, target, relu_attributions=False)
        # Upsample attributions
        return attr.LayerAttribution.interpolate(attrs, self.upsample_shape, interpolate_mode="bilinear")
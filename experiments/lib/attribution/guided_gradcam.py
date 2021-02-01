from torch import nn
from captum import attr


class GuidedGradCAM:
    def __init__(self, model: nn.Module, last_conv_layer: nn.Module, relu_attributions=True):
        self.gc = attr.LayerGradCam(model, last_conv_layer)
        self.gbp = attr.GuidedBackprop(model)
        self.relu_attributions = relu_attributions

    def __call__(self, x, target):
        # Comput GBP attributions
        gbp_attrs = self.gbp.attribute(x, target)
        # Compute attributions
        gc_attrs = self.gc.attribute(x, target, relu_attributions=self.relu_attributions)
        # Upsample attributions
        upsampled = attr.LayerAttribution.interpolate(gc_attrs, x.shape[-2:], interpolate_mode="bilinear")
        # GradCAM aggregates over channels, check if we need to duplicate attributions in order to match input shape
        if upsampled.shape[1] != x.shape[1]:
            upsampled = upsampled.repeat(1, x.shape[1], 1, 1)
        return gbp_attrs * upsampled
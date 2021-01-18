from torch import nn
from captum import attr


class GuidedGradCAM:
    def __init__(self, model: nn.Module, last_conv_layer: nn.Module):
        self.method = attr.GuidedGradCam(model, last_conv_layer)

    def __call__(self, x, target):
        # Interpolation in GGC paper is bilinear,
        # default value in Captum is nearest
        return self.method.attribute(x, target, interpolate_mode="bilinear")

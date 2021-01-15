from torch import nn
from captum import attr


class GuidedGradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self.method = attr.GuidedGradCam(self.model, self.layer)

    def __call__(self, x, target):
        # Interpolation in GGC paper is bilinear,
        # default value in Captum is nearest
        return self.method.attribute(x, target, interpolate_mode="bilinear")

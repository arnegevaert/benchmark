from experiments.lib.attribution import GradCAM
from torch import nn
from captum import attr


class GuidedGradCAM:
    """
    GuidedGradCAM is just element-wise product of guided backprop and gradCAM.
    Captum implementation multiplies with only non-negative elements of gradCAM which can result in constant
    zero saliency maps.
    """
    def __init__(self, model: nn.Module, layer: nn.Module, upsample_shape):
        super().__init__()
        self.model = model
        self.layer = layer
        self.gbp = attr.GuidedBackprop(model)
        self.gcam = GradCAM(model, layer, upsample_shape)

    def __call__(self, x, target):
        gcam_attrs = self.gcam(x, target)
        gbp_attrs = self.gbp.attribute(x, target)
        return gcam_attrs * gbp_attrs
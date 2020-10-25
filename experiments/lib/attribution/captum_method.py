from captum import attr
import torch.nn as nn


class CaptumMethod:
    def __init__(self, method: attr.Attribution):
        self.method = method

    def __call__(self, x, target):
        return self.method.attribute(x, target=target)


class Gradient(CaptumMethod):
    def __init__(self, forward_func):
        super(Gradient, self).__init__(attr.Saliency(forward_func))


class Deconvolution(CaptumMethod):
    def __init__(self, model: nn.Module):
        super(Deconvolution, self).__init__(attr.Deconvolution(model))


class GuidedBackprop(CaptumMethod):
    def __init__(self, model: nn.Module):
        super(GuidedBackprop, self).__init__(attr.GuidedBackprop(model))


class InputXGradient(CaptumMethod):
    def __init__(self, forward_func):
        super(InputXGradient, self).__init__(attr.InputXGradient(forward_func))

class SmoothGrad(CaptumMethod):
    def __init__(self, forward_func):
        method = attr.NoiseTunnel(attr.Saliency(forward_func))
        super(SmoothGrad, self).__init__(method)
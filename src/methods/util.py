from methods import *

METHODS = {
    "GuidedGradCAM": lambda model, **kwargs: GuidedGradCAM(model, **kwargs),
    "Gradient": lambda model, **kwargs: SimpleCaptumMethod(model.net, "Gradient", **kwargs),
    "InputXGradient": lambda model, **kwargs: SimpleCaptumMethod(model.net, "InputXGradient", **kwargs),
    "IntegratedGradients": lambda model, **kwargs: SimpleCaptumMethod(model.net, "IntegratedGradients", **kwargs),
    "GuidedBackprop": lambda model, **kwargs: SimpleCaptumMethod(model.net, "GuidedBackprop", **kwargs),
    "Deconvolution": lambda model, **kwargs: SimpleCaptumMethod(model.net, "Deconvolution", **kwargs),
    "Occlusion": lambda model, **kwargs: Occlusion(model.net, **kwargs),
    "Ablation": lambda model, **kwargs: SimpleCaptumMethod(model.net, "Ablation", **kwargs),
    "Random": lambda _, **kwargs: Random(**kwargs)
}


def get_all_method_constructors(include_random=True):
    if include_random:
        return METHODS
    return {k: METHODS[k] for k in METHODS.keys() if not k == "Random"}


def get_method_constructors(names=None):
    if names:
        return {k: METHODS[k] for k in names}
    return METHODS


def get_method(name, model):
    return METHODS[name](model)

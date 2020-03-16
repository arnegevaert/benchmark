from methods import *

METHODS = {
    "Gradient": lambda model: Gradient(model.net),
    "InputXGradient": lambda model: InputXGradient(model.net),
    "IntegratedGradients": lambda model: IntegratedGradients(model.net),
    "GuidedBackprop": lambda model: GuidedBackprop(model.net),
    #"GuidedGradCAM": lambda model: GuidedGradCAM(model.net),
    "Deconvolution": lambda model: Deconvolution(model.net),
    "Random": lambda _: Random()
}


def get_all_method_constructors(include_random=True):
    if include_random:
        return METHODS
    return {k: METHODS[k] for k in METHODS.keys() if not k == "Random"}


def get_method_constructors(names=None):
    if names:
        return {k: METHODS[k] for k in names}
    return METHODS
from methods import *

METHODS = {
    #"Gradient": lambda model: Gradient(model.net),
    #"InputXGradient": lambda model: InputXGradient(model.net),
    #"IntegratedGradients": lambda model: IntegratedGradients(model.net),
    #"GuidedBackprop": lambda model: GuidedBackprop(model.net),
    "GuidedGradCAM": lambda model: GuidedGradCAM(model),
    #"Deconvolution": lambda model: Deconvolution(model.net),
    #"Ablation": lambda model: Ablation(model.net),
    "Gradient": lambda model: SimpleCaptumMethod(model.net, "Gradient"),
    "InputXGradient": lambda model: SimpleCaptumMethod(model.net, "InputXGradient"),
    "IntegratedGradients": lambda model: SimpleCaptumMethod(model.net, "IntegratedGradients"),
    "GuidedBackprop": lambda model: SimpleCaptumMethod(model.net, "GuidedBackprop"),
    "Deconvolution": lambda model: SimpleCaptumMethod(model.net, "Deconvolution"),
    "Ablation": lambda model: SimpleCaptumMethod(model.net, "Ablation"),
    "Occlusion": lambda model: Occlusion(model.net),
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
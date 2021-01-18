from experiments.lib.attribution import *


_CAPTUM_METHODS = {
    "Gradient": lambda m: attr.Saliency(m),
    "SmoothGrad": lambda m: SmoothGrad(m),
    "InputXGradient": lambda m: attr.InputXGradient(m),
    "GuidedBackprop": lambda m: attr.GuidedBackprop(m),
    "Deconvolution": lambda m: attr.Deconvolution(m)
}

_UPSAMPLE_METHODS = {
    "GradCAM": lambda m, shape: GradCAM(m, m.get_last_conv_layer(), shape),
    "GuidedGradCAM": lambda m, shape: GuidedGradCAM(m, m.get_last_conv_layer()),
}

_BASELINE_METHODS = {
    "Random": lambda: Random(),
    "EdgeDetection": lambda: EdgeDetection(),
}

# This might be useful for smoothing methods as well
_INTERNAL_BS_METHODS = {
    "IntegratedGradients": lambda m, bs: IntegratedGradients(m, internal_batch_size=bs),
}

def get_methods(model, aggregation_fn, normalize, methods=None, batch_size=None, sample_shape=None):
    def _instantiate(m_name):
        if m_name in _CAPTUM_METHODS:
            # Methods from captum use .attribute() instead of .__call__()
            method_obj = _CAPTUM_METHODS[m_name](model)
            return lambda x, target: method_obj.attribute(x, target=target)
        if m_name in _UPSAMPLE_METHODS:
            # Upsampling methods need an extra argument for original shape of sample
            return _UPSAMPLE_METHODS[m_name](model, sample_shape)
        if m_name in _BASELINE_METHODS:
            # Baseline methods take no arguments
            return _BASELINE_METHODS[m_name]()
        if m_name in _INTERNAL_BS_METHODS:
            # Some methods need an internal batch size argument
            return _INTERNAL_BS_METHODS[m_name](model, batch_size)
    # Instantiate base methods
    all_keys = list(_CAPTUM_METHODS.keys()) + list(_UPSAMPLE_METHODS.keys()) +\
               list(_BASELINE_METHODS.keys()) + list(_INTERNAL_BS_METHODS.keys())
    keys = methods if methods else all_keys
    method_objs = {key: _instantiate(key) for key in keys}
    # Add aggregation wrappers if necessary
    if aggregation_fn:
        method_objs = {key: PixelAggregation(method_objs[key], aggregation_fn) for key in method_objs}
    return method_objs


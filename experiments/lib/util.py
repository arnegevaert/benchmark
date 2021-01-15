from experiments.lib.attribution.guided_gradcam import GuidedGradCAM
from experiments.lib.attribution.smooth_grad import SmoothGrad
from experiments.lib.datasets import Cifar, MNIST, ImageNette, Aptos, Cxr8, CBIS_DDSM_patches
from experiments.lib.models import Alexnet, BasicCNN, BasicMLP, Densenet, Mobilenet_v2, Resnet, Squeezenet, Vgg
from experiments.lib.attribution import *
from os import path
import os

_DATA_LOC = os.environ["BM_DATA_LOC"] if "BM_DATA_LOC" in os.environ else path.join(path.dirname(__file__), "../../data")

_DATASET_MODELS = {
    "MNIST": {
        "ds": lambda: MNIST(path.join(_DATA_LOC, "MNIST"), train=False),
        "n_pixels": 28*28,
        "models": {
            "CNN": lambda: BasicCNN(10, path.join(_DATA_LOC, "models/MNIST/cnn.pt"))
        }
    },
    "CIFAR10": {
        "ds": lambda: Cifar(path.join(_DATA_LOC, "CIFAR10"), train=False),
        "n_pixels": 32*32,
        "models": {
            "resnet18": lambda: Resnet("resnet18", 10, path.join(_DATA_LOC, "models/CIFAR10/resnet18.pt"))
        }
    },
    "ImageNette": {
        "ds": lambda: ImageNette(path.join(_DATA_LOC, "imagenette2"), train=False),
        "n_pixels": 224*224,
        "models": {
            "alexnet": lambda: Alexnet(10, path.join(_DATA_LOC, "models/ImageNette/alexnet.pt")),
            "densenet": lambda: Densenet("densenet121", 10, path.join(_DATA_LOC, "models/ImageNette/densenet121.pt")),
            "mobilenet_v2": lambda: Mobilenet_v2(10, path.join(_DATA_LOC, "models/ImageNette/mobilenet_v2.pt")),
            "resnet18": lambda: Resnet("resnet18", 10, path.join(_DATA_LOC, "models/ImageNette/resnet18.pt")),
            "squeezenet1_0": lambda: Squeezenet("squeezenet1_0.pt", 10, path.join(_DATA_LOC, "models/ImageNette/squeezenet1_0.pt")),
            "vgg11_bn": lambda: Vgg("vgg11_bn", 10, path.join(_DATA_LOC, "models/ImageNette/vgg11_bn.pt"))
        }
    },
    "Aptos": {
        "ds": lambda: Aptos(path.join(_DATA_LOC, "APTOS"), train=False),
        "n_pixels": 224*224,
        "models": {
            "densenet121": lambda: Densenet("densenet121", 5, path.join(_DATA_LOC, "models/Aptos/densenet121.pt"))
        }
    },
    "CXR8": {
        "ds": lambda: Cxr8(path.join(_DATA_LOC, "CXR8"), train=False),
        "n_pixels": 512*512,
        "models": {
            "densenet121": lambda: Densenet("densenet121", 14, path.join(_DATA_LOC, "models/CXR8/densenet121.pt"))
        }
    },
    "CBIS-DDSM": {
        "ds": lambda :CBIS_DDSM_patches(path.join(_DATA_LOC,"CBIS-DDSM"), train=False,imsize=224),
        "n_pixels": 224*224,
        "models":{
            "resnet50": lambda: Resnet("resnet50", 2, path.join(_DATA_LOC,"models/CBIS-DDSM/resnet50.pt"))
        }
    }
}

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


def get_ds_model(dataset, model):
    if dataset not in _DATASET_MODELS:
        raise ValueError(f"Dataset {dataset} not found.")
    ds_data = _DATASET_MODELS[dataset]
    if model not in ds_data["models"]:
        raise ValueError(f"Model {model} not found for {dataset}.")
    ds_obj = ds_data["ds"]()
    model_obj = ds_data["models"][model]()
    return ds_obj, model_obj


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
    # Add normalization wrappers if necessary
    if normalize:
        method_objs = {key: Normalization(method_objs[key]) for key in method_objs}
    return method_objs


def get_n_pixels(dataset):
    return _DATASET_MODELS[dataset]["n_pixels"]

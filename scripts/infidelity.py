from os import path
import torch
import argparse
import itertools
import numpy as np
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench import datasets, attribution, models
from attrbench.evaluation.infidelity import infidelity

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str)
parser.add_argument("--model-params", type=str)
parser.add_argument("--model-version", type=str, default=None)
parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette"])
parser.add_argument("--output-transform", type=str, choices=["identity", "softmax"])
parser.add_argument("--num-perturbations", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-batches", type=int, default=16)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--data-root", type=str, default="../data")
parser.add_argument("--output-file", type=str, default="result.json")
args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

if path.isfile(args.output_file) and path.getsize(args.output_file) > 0:
    sys.exit("Experiment output file is not empty")

with open(args.output_file, "w") as f:
    if not f.writable():
        sys.exit("Output file is not writable")

if args.dataset == "CIFAR10":
    dataset = datasets.Cifar(data_location=path.join(args.data_root, "CIFAR10"), train=False)
elif args.dataset == "MNIST":
    dataset = datasets.MNIST(data_location=path.join(args.data_root, "MNIST"), train=False)
elif args.dataset == "ImageNette":
    dataset = datasets.ImageNette(data_location=path.join(args.data_root, "imagenette2"), train=False)

model_constructor = getattr(models, args.model_type)
model_kwargs = {
    "params_loc": args.model_params,
    "output_logits": True,
    "num_classes": dataset.num_classes
}
if args.model_version:
    model_kwargs["version"] = args.model_version
model = model_constructor(**model_kwargs)
model.to(device)
model.eval()

kwargs = {
    "normalize": True,
    "aggregation_fn": "avg"
}

attribution_methods = {
    "Gradient": attribution.Gradient(model, **kwargs),
    "SmoothGrad": attribution.SmoothGrad(model, **kwargs),
    "InputXGradient": attribution.InputXGradient(model, **kwargs),
    "IntegratedGradients": attribution.IntegratedGradients(model, **kwargs),
    "GuidedBackprop": attribution.GuidedBackprop(model, **kwargs),
    "Deconvolution": attribution.Deconvolution(model, **kwargs),
    # "Ablation": attribution.Ablation(model, **kwargs),
    "GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs),
    "GradCAM": attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
}

dataloader = itertools.islice(DataLoader(dataset, batch_size=args.batch_size), args.num_batches)
result = infidelity(dataloader, model, attribution_methods, list(np.linspace(0.05, 0.3, 6)),
                    args.num_perturbations, True, device, args.output_transform)
result.save_json(args.output_file)

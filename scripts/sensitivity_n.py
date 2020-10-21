from os import path
import torch
import argparse
import itertools
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench import datasets, attribution, models
from attrbench.evaluation.sensitivity_n import sensitivity_n

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str)
parser.add_argument("--model-params", type=str)
parser.add_argument("--model-version", type=str, default=None)
parser.add_argument("--num-subsets", type=int, default=100)  # Default value from paper
parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette", "Aptos"])
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-batches", type=int, default=16)
parser.add_argument("--output-transform", type=str, choices=["identity", "softmax"])
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
    mask_range = list(range(0, 32*32, 30))
elif args.dataset == "MNIST":
    dataset = datasets.MNIST(data_location=path.join(args.data_root, "MNIST"), train=False)
    mask_range = list(range(0, 28*28, 25))
elif args.dataset == "ImageNette":
    dataset = datasets.ImageNette(data_location=path.join(args.data_root, "imagenette2"), train=False)
    mask_range = list(range(0, 224*224, 4000))
elif args.dataset == "Aptos":
    dataset = datasets.Aptos(img_size=320, data_location=path.join(args.data_root, "APTOS"), train=False)
    mask_range = list(range(6800, 320*320, 6800))

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

dataloader = itertools.islice(DataLoader(dataset, batch_size=args.batch_size, num_workers=4), args.num_batches)
result = sensitivity_n(dataloader, model,
                       attribution_methods, mask_range=mask_range,
                       n_subsets=args.num_subsets, mask_value=dataset.mask_value,
                       pixel_level_mask=True, device=device, output_transform=args.output_transform)
result.save_json(args.output_file)
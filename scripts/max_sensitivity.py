import pandas as pd
import numpy as np
from os import path
import torch
import argparse
import json

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench import datasets, attribution, models
from attrbench.evaluation.max_sensitivity import max_sensitivity

parser = argparse.ArgumentParser()
parser.add_argument("--perturb-min", type=float)
parser.add_argument("--perturb-max", type=float)
parser.add_argument("--perturb-amt", type=int)
parser.add_argument("--model-type", type=str)
parser.add_argument("--model-params", type=str)
parser.add_argument("--model-version", type=str, default=None)
parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette"], default="MNIST")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--use-logits", type=bool, default=True)
parser.add_argument("--normalize-attrs", type=bool, default=True)
parser.add_argument("--aggregation-fn", type=str, choices=["avg", "max-abs"], default="avg")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--data-root", type=str, default="../data")
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--out-dir", type=str, default="../out")
parser.add_argument("--allow-overwrite", action="store_true")
args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

if path.isfile(path.join(args.out_dir, f"{args.experiment_name}.pkl")) and not args.allow_overwrite:
    exit("Experiment output file already exists. Allow overwriting experiment files or choose a different experiment name.")

if args.dataset == "CIFAR10":
    dataset = datasets.Cifar(batch_size=args.batch_size, data_location=path.join(args.data_root, "CIFAR10"),
                             download=False, shuffle=True, version="cifar10")
elif args.dataset == "MNIST":
    dataset = datasets.MNIST(batch_size=args.batch_size, data_location=path.join(args.data_root, "MNIST"),
                             download=False, shuffle=True)
elif args.dataset == "ImageNette":
    dataset = datasets.ImageNette(batch_size=args.batch_size, data_location=path.join(args.data_root, "ImageNette"),
                                  shuffle=True)

perturbation_range = list(np.linspace(args.perturb_min, args.perturb_max, args.perturb_amt))
print(perturbation_range)
model_constructor = getattr(models, args.model_type)
model_kwargs = {
    "params_loc": args.model_params,
    "output_logits": args.use_logits,
    "num_classes": dataset.num_classes
}
if args.model_version:
    model_kwargs["version"] = args.model_version
model = model_constructor(**model_kwargs)
model.to(device)
model.eval()

kwargs = {
    "normalize": args.normalize_attrs,
    "aggregation_fn": args.aggregation_fn
}

attribution_methods = {
    "Gradient": attribution.Gradient(model, **kwargs),
    "SmoothGrad": attribution.SmoothGrad(model, **kwargs),
    "InputXGradient": attribution.InputXGradient(model, **kwargs),
    "IntegratedGradients": attribution.IntegratedGradients(model, **kwargs),
    "GuidedBackprop": attribution.GuidedBackprop(model, **kwargs),
    "Deconvolution": attribution.Deconvolution(model, **kwargs),
    "Ablation": attribution.Ablation(model, **kwargs),
    "GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), **kwargs),
    "GradCAM": attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
}

result = max_sensitivity(dataset.get_dataloader(train=False), attribution_methods,
                         perturbation_range, device)

result_df = pd.DataFrame.from_dict(result).stack().reset_index()
result_df.columns = ["r", "method", "max_sens"]
result_df["r"] = np.array(perturbation_range)[result_df["r"]]

result_df.to_pickle(path.join(args.out_dir, f"{args.experiment_name}.pkl"))
meta_filename = path.join(args.out_dir, f"{args.experiment_name}_args.json")
with open(meta_filename, "w") as f:
    json.dump(vars(args), f)

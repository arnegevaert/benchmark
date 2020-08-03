import numpy as np
import pandas as pd
from os import path
import torch
import argparse
import json
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench import attribution, models
from attrbench.evaluation.bam import input_dependence_rate, BAMDataset

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str, default="Resnet")
parser.add_argument("--model-params", type=str, default="../../data/models/BAM/scene/resnet18.pt")
parser.add_argument("--model-version", type=str, default="resnet18")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--use-logits", type=bool, default=True)
# TODO normalizing might not be logical here as we are comparing across "different" samples
parser.add_argument("--normalize-attrs", type=bool, default=True)
parser.add_argument("--aggregation-fn", type=str, choices=["avg", "max-abs"], default="avg")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--data-root", type=str, default="../../data")
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--out-dir", type=str, default="../../out/BAM/IDR")
args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

if not path.isdir(args.out_dir):
    exit(f"Directory {args.out_dir} does not exist")

if path.isfile(path.join(args.out_dir, f"{args.experiment_name}.pkl")):
    exit("Experiment output file already exists")

dataset = BAMDataset(path.join(args.data_root, "BAM"), train=False, include_orig_scene=True, include_mask=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

import itertools
dataloader = itertools.islice(iter(dataloader), 4)

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
    #"InputXGradient": attribution.InputXGradient(model, **kwargs),
    #"IntegratedGradients": attribution.IntegratedGradients(model, **kwargs),
    #"GuidedBackprop": attribution.GuidedBackprop(model, **kwargs),
    #"Deconvolution": attribution.Deconvolution(model, **kwargs),
    #"Ablation": attribution.Ablation(model, **kwargs),
    #"GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs),
    #"GradCAM": attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
}

result = input_dependence_rate(dataloader, model, attribution_methods, device)
result_df = pd.concat([pd.DataFrame(result[m_name]).assign(method=m_name) for m_name in attribution_methods])

result_df.to_pickle(path.join(args.out_dir, f"{args.experiment_name}.pkl"))
meta_filename = path.join(args.out_dir, f"{args.experiment_name}_args.json")
with open(meta_filename, "w") as f:
    json.dump(vars(args), f)
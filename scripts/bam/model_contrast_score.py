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
from attrbench.evaluation.bam import model_contrast_score, BAMDataset

parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str)
parser.add_argument("--object-model-params", type=str)
parser.add_argument("--scene-model-params", type=str)
parser.add_argument("--model-version", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--use-logits", type=bool, default=True)
# TODO normalizing might not be logical here as we are comparing across "different" models
parser.add_argument("--normalize-attrs", type=bool, default=True)
parser.add_argument("--aggregation-fn", type=str, choices=["avg", "max-abs"], default="avg")
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--data-root", type=str, default="../../data")
parser.add_argument("--experiment-name", type=str, default="experiment")
parser.add_argument("--out-dir", type=str, default=".")
args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

if not path.isdir(args.out_dir):
    exit(f"Directory {args.out_dir} does not exist")

if path.isfile(path.join(args.out_dir, f"{args.experiment_name}.pkl")):
    exit("Experiment output file already exists")

dataset = BAMDataset(path.join(args.data_root, "BAM"), train=False, include_orig_scene=False, include_mask=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

model_constructor = getattr(models, args.model_type)
model_kwargs = {
    "output_logits": args.use_logits,
    "num_classes": dataset.num_classes
}
if args.model_version:
    model_kwargs["version"] = args.model_version

object_model = model_constructor(**{**model_kwargs, **{"params_loc": args.object_model_params}})
object_model.to(device)
object_model.eval()

scene_model = model_constructor(**{**model_kwargs, **{"params_loc": args.scene_model_params}})
scene_model.to(device)
scene_model.eval()

kwargs = {
    "normalize": args.normalize_attrs,
    "aggregation_fn": args.aggregation_fn
}


attribution_methods = {
    "Gradient": lambda model: attribution.Gradient(model, **kwargs),
    "SmoothGrad": lambda model: attribution.SmoothGrad(model, **kwargs),
    #"InputXGradient": lambda model: attribution.InputXGradient(model, **kwargs),
    #"IntegratedGradients": lambda model: attribution.IntegratedGradients(model, **kwargs),
    #"GuidedBackprop": lambda model: attribution.GuidedBackprop(model, **kwargs),
    #"Deconvolution": lambda model: attribution.Deconvolution(model, **kwargs),
    #"Ablation": lambda model: attribution.Ablation(model, **kwargs),
    #"GuidedGradCAM": lambda model: attribution.GuidedGradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs),
    #"GradCAM": lambda model: attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
}
object_methods = {m_name: attribution_methods[m_name](object_model) for m_name in attribution_methods}
scene_methods = {m_name: attribution_methods[m_name](scene_model) for m_name in attribution_methods}

result = model_contrast_score(dataloader, object_methods, scene_methods, device)

result_df = pd.concat([pd.DataFrame(result[m_name]).assign(method=m_name) for m_name in attribution_methods])
result_df.to_pickle(path.join(args.out_dir, f"{args.experiment_name}.pkl"))
meta_filename = path.join(args.out_dir, f"{args.experiment_name}_args.json")
with open(meta_filename, "w") as f:
    json.dump(vars(args), f)
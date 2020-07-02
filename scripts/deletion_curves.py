from attrbench import datasets, attribution, models
from attrbench.evaluation.deletion_curves import deletion_curves
import numpy as np
import pandas as pd
from os import path
import torch
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("mask-max", type=int)
parser.add_argument("mask-interval", type=int)
parser.add_argument("dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette"], default="CIFAR10")
parser.add_argument("model-type", type=str, default="Alexnet")
parser.add_argument("model-version", type=str, default=None)
parser.add_argument("batch-size", type=int, default=64)
parser.add_argument("use-logits", type=bool, default=True)
parser.add_argument("normalize-attrs", type=bool, default=True)
parser.add_argument("aggregation-fn", type=str, choices=["avg", "max-abs"], default="avg")
parser.add_argument("cuda", type=bool, default=True)
parser.add_argument("data-root", type=str)
parser.add_argument("experiment-name", type=str, default="experiment")
args = parser.parse_args()
device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

if not path.isfile(path.join(args.data_root, "models", args.dataset, f"{args.model}.pt")):
    sys.exit(f"Model {args.model} for dataset {args.dataset} not found in {args.data_root}")

if args.dataset_name == "CIFAR10":
    dataset = datasets.Cifar(batch_size=args.batch_size, data_location=path.join(args.data_root, "CIFAR10"),
                             download=False, shuffle=True, version="cifar10")
elif args.dataset_name == "MNIST":
    dataset = datasets.MNIST(batch_size=args.batch_size, data_location=path.join(args.data_root, "MNIST"),
                             download=False, shuffle=True)
elif args.dataset_name == "ImageNette":
    dataset = datasets.ImageNette(batch_size=args.batch_size, data_location=path.join(args.data_root, "ImageNette"),
                                  shuffle=True)

mask_range = list(range(1, args.mask_max, args.mask_interval))
model_constructor = getattr(models, args.model_type)
model_kwargs = {
    "batch_size": args.batch_size,
    "params_loc": path.join(args.data_root, f"models/{args.dataset}/{args.model}.pt"),
    "output_logits": args.use_logits,
    "num_classes": dataset.num_classes
}
if args.model_version:
    model_kwargs["model_version"] = args.model_version
model = model_constructor(**model_kwargs)
model.to(device)
model.eval()


kwargs = {
    "normalize": args.normalize_attrs,
    "aggregation_fn": args.aggregation_fn
}

attribution_methods = {
    #"GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), **kwargs),
    "Gradient": attribution.Gradient(model, **kwargs),
    "InputXGradient": attribution.InputXGradient(model, **kwargs),
    "IntegratedGradients": attribution.IntegratedGradients(model, **kwargs),
    "GuidedBackprop": attribution.GuidedBackprop(model, **kwargs),
    "Deconvolution": attribution.Deconvolution(model, **kwargs),
}

result = deletion_curves(dataset.get_dataloader(train=False), model,
                         attribution_methods, mask_range, dataset.mask_value,
                         pixel_level_mask=args.aggregation_fn is not None, device=device)

result_df = pd.DataFrame.from_dict(
    {m_name: pd.DataFrame(data=result[m_name]).stack() for m_name in attribution_methods}
).stack().reset_index()
result_df.columns = ["sample", "mask", "method", "difference"]
result_df["mask"] = np.array(mask_range)[result_df["mask"]]



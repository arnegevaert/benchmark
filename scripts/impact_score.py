import numpy as np
import pandas as pd
from os import path
import torch
import argparse
import json
import itertools
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
from attrbench import datasets, attribution, models
from attrbench.evaluation.impact_score import impact_score

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="BasicCNN")
    parser.add_argument("--model-params", type=str, default="../data/models/MNIST/cnn.pt")
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette"], default="MNIST")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--i-score-output-file", type=str, default="i_score.json")
    parser.add_argument("--strict-i-score-output-file", type=str, default="strict_i_score.json")
    args = parser.parse_args()
    return args


def main(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    for outfile in [args.i_score_output_file, args.strict_i_score_output_file]:
        if path.isfile(outfile) and path.getsize(outfile) > 0:
            sys.exit("Experiment output file is not empty")

        with open(outfile, "w") as f:
            if not f.writable():
                sys.exit("Output file is not writable")

    if args.dataset == "CIFAR10":
        dataset = datasets.Cifar(data_location=path.join(args.data_root, "CIFAR10"), train=False)
        mask_range = list(range(30, 32 * 32, 30))
    elif args.dataset == "MNIST":
        dataset = datasets.MNIST(data_location=path.join(args.data_root, "MNIST"), train=False)
        mask_range = list(range(25, 28 * 28, 25))
    elif args.dataset == "ImageNette":
        dataset = datasets.ImageNette(data_location=path.join(args.data_root, "imagenette2"), train=False)
        mask_range = list(range(4000, 224 * 224, 4000))

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
        "normalize": False,  # Normalizing isn't necessary, only order of values counts
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
        "GuidedGradCAM": attribution.GuidedGradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:],
                                                   **kwargs),
        "GradCAM": attribution.GradCAM(model, model.get_last_conv_layer(), dataset.sample_shape[1:], **kwargs)
    }

    dataloader = itertools.islice(DataLoader(dataset, batch_size=args.batch_size, num_workers=4), args.num_batches)
    i_score, i_strict_score = impact_score(dataloader, model, list(mask_range), methods=attribution_methods,
                                           mask_value=dataset.mask_value, tau=args.tau, device=device)
    i_score.save_json(args.i_score_output_file)
    i_strict_score.save_json(args.strict_i_score_output_file)


if __name__ == '__main__':  # windows machines do weird stuff when there is no main guard
    args = parse_args()
    main(args)

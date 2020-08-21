from os import path
import torch
import argparse
import itertools
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench import datasets, attribution, models
from attrbench.evaluation.impact_coverage import impact_coverage



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="Resnet")
    parser.add_argument("--model-params", type=str, default="../../data/models/CIFAR10/resnet18.pt")
    parser.add_argument("--model-version", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette"], default="CIFAR10")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--data-root", type=str, default="../../data")
    parser.add_argument("--output-file", type=str, default="result.json")
    args = parser.parse_args()
    return args


def main(args):
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
        "normalize": False,
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

    if args.model_version:
        patch_location = path.join(args.data_root, "patches",
                                   f"{args.dataset}_{args.model_type}_{args.model_version}_{args.target_label}_patch.pt")
    else:
        patch_location = path.join(args.data_root, "patches",
                                   f"{args.dataset}_{args.model_type}_{args.target_label}_patch.pt")

    patch = torch.load(patch_location)
    dl = itertools.islice(DataLoader(dataset, batch_size=args.batch_size), args.num_batches)
    result = impact_coverage(dl, patch=patch, model=model, methods=attribution_methods,
                             device=device, target_label=args.target_label)
    result.save_json(args.output_file)


if __name__ == '__main__':  # windows machines do weird stuff when there is no main guard
    args = parse_args()
    main(args)

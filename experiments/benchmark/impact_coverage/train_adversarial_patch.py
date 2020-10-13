import argparse
import torch
from os import path
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import os
import sys
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from experiments.lib import datasets, models
from impact_coverage import make_patch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="Resnet")
    parser.add_argument("--model-params", type=str, default="../../data/models/ImageNette/resnet18.pt")
    parser.add_argument("--model-version", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette", "Aptos"], default="ImageNette")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--data-root", type=str, default="../../data")
    parser.add_argument("--out-dir", type=str, default="../../data/patches")
    parser.add_argument("--patch-percent", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=20)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.dataset == "CIFAR10":
        dataset = datasets.Cifar(data_location=path.join(args.data_root, "CIFAR10"), train=False)
    elif args.dataset == "MNIST":
        dataset = datasets.MNIST(data_location=path.join(args.data_root, "MNIST"), train=False)
    elif args.dataset == "ImageNette":
        dataset = datasets.ImageNette(data_location=path.join(args.data_root, "imagenette2"), train=False)
    elif args.dataset == "Aptos":
        dataset = datasets.Aptos(data_location=path.join(args.data_root, "APTOS"), img_size=320, train=False)

    model_kwargs = {
        "params_loc": args.model_params,
        "output_logits": True,
        "num_classes": dataset.num_classes
    }
    model_constructor = getattr(models, args.model_type)
    if args.model_version:
        model_kwargs["version"] = args.model_version
    model = model_constructor(**model_kwargs)
    model.to(device)
    model.eval()

    if args.model_version:
        patch_file = f"{args.dataset}_{args.model_type}_{args.model_version}_{args.target_label}_patch.pt"
    else:
        patch_file = f"{args.dataset}_{args.model_type}_{args.target_label}_patch.pt"

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    make_patch(dataloader, model, args.target_label,
               path.join(args.out_dir, patch_file), device,
               epochs=args.epochs,
               lr=args.lr, patch_percent=args.patch_percent)

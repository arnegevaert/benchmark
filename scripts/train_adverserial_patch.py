import argparse
import torch
from os import path
from attrbench import datasets, attribution, models
from attrbench.evaluation.impact_coverage import make_patch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str)
    parser.add_argument("--model-params", type=str)
    parser.add_argument("--model-version", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "ImageNette"], default="MNIST")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-logits", type=bool, default=True)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--data-root", type=str, default="../data")
    parser.add_argument("--out-dir", type=str, default="../data/patches")
    parser.add_argument("--patch-percent", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=20)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.dataset == "CIFAR10":
        dataset = datasets.Cifar(batch_size=args.batch_size, data_location=path.join(args.data_root, "CIFAR10"),
                                 download=False, shuffle=True, version="cifar10")
    elif args.dataset == "MNIST":
        dataset = datasets.MNIST(batch_size=args.batch_size, data_location=path.join(args.data_root, "MNIST"),
                                 download=False, shuffle=True)
    elif args.dataset == "ImageNette":
        dataset = datasets.ImageNette(batch_size=args.batch_size, data_location=path.join(args.data_root, "ImageNette"),
                                      shuffle=True)
    elif args.dataset == "Aptos":
        dataset = datasets.Aptos(batch_size=args.batch_size, data_location=path.join(args.data_root, "APTOS"),
                                 img_size=320)
    model_kwargs = {
        "params_loc": args.model_params,
        "output_logits": args.use_logits,
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

    make_patch(dataset.get_dataloader(train=False), model, args.target_label,
               path.join(args.out_dir, patch_file), device,
               epochs=args.epochs,
               lr=args.lr, patch_percent=args.patch_percent)

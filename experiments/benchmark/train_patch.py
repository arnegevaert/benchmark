import argparse
import torch
from experiments.lib.util import get_ds_model
from attrbench.util import make_patch
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-s", "--patch_size", type=float, default=0.05)
    parser.add_argument("-t", "--target", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # Try opening output file to make sure location is writable
    with open(args.output, "wb") as f:
        print("Output location OK")
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    dataset, model = get_ds_model(args.dataset, args.model)
    model.to(device)
    model.eval()

    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    make_patch(dl, model, args.target, args.output, device, epochs=args.epochs,
               lr=args.lr, patch_percent=args.patch_size)
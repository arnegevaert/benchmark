import argparse
import torch
import os
from os import path
from torch import optim, nn
from torch.utils.data import DataLoader

# This block allows us to import from the benchmark folder,
# as if it was a package installed using pip
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from attrbench import models
from attrbench.evaluation.bam import train_epoch, BAMDataset

"""
Trains a model on one of the BAM datasets.
MCS: requires model trained on scene labels and model trained on object labels
IDR: requires model trained on scene labels
IIR: requires model trained on original scene images (without object overlays)
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="../../data")
    parser.add_argument("--labels", type=str, choices=["obj", "scene"], default="obj")
    parser.add_argument("--output-dir", type=str, default="out")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load-checkpoint", type=bool, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Construct model
    #model = models.Resnet("resnet18", output_logits=True, num_classes=10)
    model = models.Alexnet(output_logits=True, num_classes=10)
    start_epoch = 0
    if args.load_checkpoint:
        checkpoints = os.listdir(args.output_dir)
        if len(checkpoints) > 0:
            checkpoint_epochs = sorted([int(cp[len("checkpoint_ep"):-3]) for cp in checkpoints])
            latest = f"checkpoint_ep{checkpoint_epochs[-1]}.pt"
            print(f"Loading checkpoint: {latest}")
            model.load_state_dict(torch.load(path.join(args.output_dir, latest)))
            start_epoch = int(checkpoint_epochs[-1])
        else:
            print("No checkpoints found")
    model.to(args.device)

    # Construct dataset
    train_ds = BAMDataset(path.join(args.data_root, "BAM"), train=True)
    test_ds = BAMDataset(path.join(args.data_root, "BAM"), train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}")
        train_epoch(model, train_dl, test_dl, args.labels, optimizer, loss, args.device)
        if (epoch+1) % 5 == 0:
            filename = path.join(args.output_dir, f"checkpoint_ep{epoch+1}.pt")
            print(f"Saving {filename}")
            torch.save(model.state_dict(), filename)
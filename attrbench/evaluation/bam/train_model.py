import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

"""
Trains a model on one of the BAM datasets.
MCS: requires model trained on scene labels and model trained on object labels
IDR: requires model trained on scene labels
IIR: requires model trained on original scene images (without object overlays)
"""


def train_epoch(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader,
                optimizer, criterion, device: str):
    prog = tqdm(train_dl, desc=f"Training")
    total_samples, correct_samples = 0, 0
    losses = []
    # Train
    for batch, labels in prog:
        batch = batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        y_pred = model(batch)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total_samples += batch.size(0)
        correct_samples += (torch.argmax(y_pred, dim=1) == labels).sum().item()
        prog.set_postfix({"loss": sum(losses)/len(losses),
                          "acc": f"{correct_samples}/{total_samples} ({100*correct_samples/total_samples})"})

    # Test
    prog = tqdm(test_dl, desc="Testing")
    total_samples, correct_samples = 0, 0
    for batch, labels in prog:
        with torch.no_grad():
            batch = batch.to(device)
            labels = labels.to(device)
            y_pred = model(batch)
            total_samples += batch.size(0)
            correct_samples += (torch.argmax(y_pred, dim=1) == labels).sum().item()
            prog.set_postfix(
                {"acc": f"{correct_samples}/{total_samples} ({100 * correct_samples / total_samples})"})

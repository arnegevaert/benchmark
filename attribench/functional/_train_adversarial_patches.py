import numpy as np
from tqdm import tqdm
import torch
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
from itertools import cycle


def _normalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def _init_patch_square(
    image_size, image_channels, patch_size_percent, data_min, data_max
):
    image_size = image_size**2
    noise_size = image_size * patch_size_percent
    noise_dim = int(noise_size**0.5)
    patch = np.random.rand(1, image_channels, noise_dim, noise_dim)
    patch = _normalize(patch, data_min, data_max)
    return patch


def _train_epoch(
    model,
    patch,
    train_dl,
    loss_function,
    optimizer,
    target_label,
    data_min,
    data_max,
    device,
):
    patch_size = patch.shape[-1]
    train_loss = []
    target = None
    for x, y in train_dl:
        # x, y = torch.tensor(x), torch.tensor(y)
        optimizer.zero_grad()
        if target is None:
            target = torch.tensor(
                np.full(y.shape[0], target_label),
                dtype=torch.long,
                device=device,
            )
        image_size = x.shape[-1]

        indx = np.random.randint(0, image_size - patch_size, size=y.shape[0])
        indy = np.random.randint(0, image_size - patch_size, size=y.shape[0])

        images = x.to(device)
        for i in range(y.shape[0]):
            images[
                i,
                :,
                indx[i] : indx[i] + patch_size,
                indy[i] : indy[i] + patch_size,
            ] = patch

        adv_out = model(images)

        loss = loss_function(adv_out, target[: y.shape[0]])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.data = torch.clamp(patch.data, min=data_min, max=data_max)
        train_loss.append(loss.item())
    epoch_loss = np.array(train_loss).mean()
    return epoch_loss


def _validate(model, patch, data_loader, loss_function, target_label, device):
    patch_size = patch.shape[-1]
    val_loss = []
    preds = []
    with torch.no_grad():
        for x, y in data_loader:
            y = torch.tensor(
                np.full(y.shape[0], target_label), dtype=torch.long
            ).to(device)
            image_size = x.shape[-1]

            indx = random.randint(0, image_size - patch_size)
            indy = random.randint(0, image_size - patch_size)

            images = x.to(device)
            images[
                :, :, indx : indx + patch_size, indy : indy + patch_size
            ] = patch
            adv_out = model(images)
            loss = loss_function(adv_out, y)

            val_loss.append(loss.item())
            preds.append(adv_out.argmax(axis=1).detach().cpu().numpy())
        val_loss = np.array(val_loss).mean()
        preds = np.concatenate(preds)
        percent_successful = (
            np.count_nonzero(preds == target_label) / preds.shape[0]
        )
        return val_loss, percent_successful


def _make_patch(
    dataset,
    batch_size,
    model,
    target_label,
    device,
    patch_percent=0.1,
    epochs=5,
    data_min=None,
    data_max=None,
    lr=0.05,
):
    print(f"Training patch for label {target_label}...")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,
                            pin_memory=True)
    # patch values will be clipped between data_min and data_max
    # so that patch will be valid image data.
    if data_max is None or data_min is None:
        for x, _ in tqdm(dataloader):
            if data_max is None:
                data_max = x.max().item()
            if data_min is None:
                data_min = x.min().item()
            if x.min() < data_min:
                data_min = x.min().item()
            if x.max() > data_max:
                data_max = x.max().item()

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    x, _ = next(iter(dataloader))
    sample_shape = x.shape

    patch = _init_patch_square(
        sample_shape[-1], sample_shape[1], patch_percent, data_min, data_max
    )
    patch = torch.tensor(patch, requires_grad=True, device=device)
    optim = torch.optim.Adam([patch], lr=lr, weight_decay=0.0)

    loss = torch.nn.CrossEntropyLoss()
    min_loss = None
    best_patch = None

    for epoch in range(epochs):
        epoch_loss = _train_epoch(
            model,
            patch,
            dataloader,
            loss,
            optim,
            target_label=target_label,
            data_min=data_min,
            data_max=data_max,
            device=device,
        )
        print(f"Patch {target_label} epoch {epoch} loss: {epoch_loss}")
        if min_loss is None or epoch_loss < min_loss:
            min_loss = epoch_loss
            best_patch = patch.cpu()

    val_loss, percent_successful = _validate(
        model, patch, dataloader, loss, target_label, device
    )
    return best_patch, val_loss, percent_successful

def train_adversarial_patches(
        model: nn.Module,
        dataset: Dataset,
        num_patches: int,
        batch_size: int,
        path: Optional[str],
        labels: Optional[Tuple[int]] = None,
        device: Optional[torch.device] = None,
) -> List[torch.Tensor] | None:
    """Train adversarial patches for a given model and dataset.
    If `path` is not `None`, the patches are saved to disk.
    Otherwise, they are returned as a list.

    Parameters
    ----------
    model : nn.Module
        The model for which the patches should be trained.
    dataset : Dataset
        Torch Dataset to use for training the patches.
    num_patches : int
        The number of patches to train.
    batch_size : int
        The batch size to use for training.
    path : Optional[str]
        The path to which the patches should be saved.
        If `None`, the patches are returned as a list.
        Default: `None`.
    labels : Optional[Tuple[int]], optional
        Tuple of labels to use for the patches.
        If `None`, the labels are assumed to be `range(num_patches)`.
        Default: `None`.
    device : Optional[torch.device], optional
        Device to use, by default None.

    Returns
    -------
    List[torch.Tensor] | None
        If `path` is `None`, a list of patches.
        Otherwise, `None`.
    """    
    if device is None:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    patch_labels = cycle(labels) if labels is not None else range(num_patches)
    all_patches = []
    for patch_label in patch_labels:
        patch, val_loss, percent_successful = _make_patch(
            dataset, batch_size, model, patch_label, device
        )
        print(
            f"Patch label: {patch_label}.",
            f"Loss: {val_loss:.3f}.",
            f"Success rate: {percent_successful:.3f}.",
        )
        if path is not None:
            torch.save(patch, path + f"_{patch_label}.pt")
        else:
            all_patches.append(patch)
    if path is None:
        return all_patches
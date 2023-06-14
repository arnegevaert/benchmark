from typing import Callable, Optional, Tuple, NoReturn
from tqdm import trange
import os
import torch
from torch import nn
import numpy as np
from torch import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import random

from attribench.distributed.distributed_computation import (
    DistributedComputation,
)
from attribench.distributed.message import PartialResultMessage
from attribench.distributed.worker import Worker


def normalize(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def init_patch_square(
    image_size, image_channels, patch_size_percent, data_min, data_max
):
    image_size = image_size**2
    noise_size = image_size * patch_size_percent
    noise_dim = int(noise_size**0.5)
    patch = np.random.rand(1, image_channels, noise_dim, noise_dim)
    patch = normalize(patch, data_min, data_max)
    return patch


def train_epoch(
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


def validate(model, patch, data_loader, loss_function, target_label, device):
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


def make_patch(
    dataloader,
    model,
    target_label,
    device,
    patch_percent=0.1,
    epochs=5,
    data_min=None,
    data_max=None,
    lr=0.05,
):
    # patch values will be clipped between data_min and data_max
    # so that patch will be valid image data.
    if data_max is None or data_min is None:
        for x, _ in dataloader:
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

    patch = init_patch_square(
        sample_shape[-1], sample_shape[1], patch_percent, data_min, data_max
    )
    patch = torch.tensor(patch, requires_grad=True, device=device)
    optim = torch.optim.Adam([patch], lr=lr, weight_decay=0.0)

    loss = torch.nn.CrossEntropyLoss()
    min_loss = None
    best_patch = None

    for i in trange(epochs):
        epoch_loss = train_epoch(
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
        print(epoch_loss)
        if min_loss is None or epoch_loss < min_loss:
            min_loss = epoch_loss
            best_patch = patch.cpu()

    val_loss, percent_successful = validate(
        model, patch, dataloader, loss, target_label, device
    )
    return best_patch, val_loss, percent_successful


class PatchResult:
    def __init__(
        self, patch_label: int, val_loss: float, percent_successful: float
    ) -> None:
        self.patch_label = patch_label
        self.val_loss = val_loss
        self.percent_successful = percent_successful


class MakePatchesWorker(Worker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: mp.Event,
        path: str,
        total_num_patches: int,
        batch_size: int,
        dataset: Dataset,
        model_factory: Callable[[], nn.Module],
        result_handler: Optional[
            Callable[[PartialResultMessage], NoReturn]
        ] = None,
    ):
        super().__init__(
            result_queue, rank, world_size, all_processes_done, result_handler
        )
        self.patch_labels = list(range(total_num_patches))[
            self.rank : total_num_patches : self.world_size
        ]
        self.dataset = dataset
        self.model_factory = model_factory
        self.batch_size = batch_size
        self.path = path

    def work(self):
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)

        for patch_label in self.patch_labels:
            # Train patch
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=4,
                pin_memory=True,
            )
            patch, val_loss, percent_successful = make_patch(
                dataloader, model, patch_label, device
            )

            # Save patch to disk
            torch.save(
                patch, os.path.join(self.path, f"patch_{patch_label}.pt")
            )

            # Send message to main process
            self.send_result(
                PartialResultMessage(
                    self.rank,
                    PatchResult(patch_label, val_loss, percent_successful),
                )
            )


class MakePatches(DistributedComputation):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: Dataset,
        num_labels: int,
        batch_size: int,
        path: str,
        address="localhost",
        port="12355",
        devices: Optional[Tuple[int]] = None,
    ):
        super().__init__(address, port, devices)
        self.num_labels = num_labels
        self.path = path
        self.prog = None
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> Worker:
        return MakePatchesWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.path,
            self.num_labels,
            self.batch_size,
            self.dataset,
            self.model_factory,
            self._handle_result if self.world_size == 1 else None,
        )

    def _handle_result(self, result: PartialResultMessage[PatchResult]):
        # The workers save the files,
        # so no need to do anything except log results
        print(
            f"Received patch {result.data.patch_label}.",
            f"Loss: {result.data.val_loss:.3f}.",
            f"Acc: {result.data.percent_successful:.3f}.",
        )

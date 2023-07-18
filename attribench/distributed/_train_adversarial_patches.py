from typing import Optional, Tuple
import os
import torch
from torch.utils.data import Dataset

from attribench.distributed._distributed_computation import (
    DistributedComputation,
)
from attribench.distributed._message import PartialResultMessage
from attribench.distributed._worker import Worker, WorkerConfig
from attribench.functional._train_adversarial_patches import _make_patch
from attribench import ModelFactory


class PatchResult:
    def __init__(
        self, patch_label: int, val_loss: float, percent_successful: float
    ) -> None:
        self.patch_label = patch_label
        self.val_loss = val_loss
        self.percent_successful = percent_successful


class AdversarialPatchTrainingWorker(Worker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        path: str,
        total_num_patches: int,
        batch_size: int,
        dataset: Dataset,
        model_factory: ModelFactory,
        labels: Optional[Tuple[int]] = None,
    ):
        super().__init__(worker_config)
        # Create a list of patch labels.
        # If the number of available labels is smaller than the number of
        # patches, the labels are repeated.
        if labels is None:
            labels = tuple(range(total_num_patches))
        num_repeats = total_num_patches // len(labels)
        labels = labels * (num_repeats + 1)

        # Each worker only trains a subset of the patches.
        rank = self.worker_config.rank
        world_size = self.worker_config.world_size
        self.patch_labels = labels[
            rank : total_num_patches : world_size
        ]
        self.dataset = dataset
        self.model_factory = model_factory
        self.batch_size = batch_size
        self.path = path

    def work(self):
        device = torch.device(self.worker_config.rank)
        model = self.model_factory()
        model.to(device)

        for patch_label in self.patch_labels:
            # Train patch
            patch, val_loss, percent_successful = _make_patch(
                self.dataset, self.batch_size, model, patch_label, device
            )

            # Save patch to disk
            torch.save(
                patch, os.path.join(self.path, f"patch_{patch_label}.pt")
            )

            # Send message to main process
            self.worker_config.send_result(
                PartialResultMessage(
                    self.worker_config.rank,
                    PatchResult(patch_label, val_loss, percent_successful),
                )
            )


class TrainAdversarialPatches(DistributedComputation):
    """Train adversarial patches for a given model and dataset and save
    them to disk. The patches are trained in parallel on multiple
    processes. Each process trains a subset of the patches.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        dataset: Dataset,
        num_patches: int,
        batch_size: int,
        path: str,
        labels: Optional[Tuple[int]] = None,
        address: str = "localhost",
        port: str = "12355",
        devices: Optional[Tuple[int]] = None,
    ):
        """
        Parameters
        ----------
        model_factory : ModelFactory
            ModelFactory instance or callable that returns a model.
            Used to create a model for each subprocess.
        dataset : Dataset
            Torch Dataset to use for training the patches.
        num_patches : int
            Number of patches to train.
        batch_size : int
            Batch size per subprocess to use for training.
        path : str
            Path to which the patches should be saved.
        labels : Optional[Tuple[int]], optional
            Tuple of labels to use for the patches.
            If `None`, the labels are assumed to be `range(num_patches)`.
            Default: `None`.
        address : str, optional
            Address for communication between subprocesses,
            by default "localhost"
        port : str, optional
            Port for communication between subprocesses, by default "12355"
        devices : Optional[Tuple[int]], optional
            Devices to use. If None, then all available devices are used.
            By default None.
        """
        super().__init__(address, port, devices)
        self.num_patches = num_patches
        self.labels = labels
        self.path = path
        self.prog = None
        self.model_factory = model_factory
        self.dataset = dataset
        self.batch_size = batch_size
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def _create_worker(self, worker_config: WorkerConfig) -> Worker:
        return AdversarialPatchTrainingWorker(
            worker_config,
            self.path,
            self.num_patches,
            self.batch_size,
            self.dataset,
            self.model_factory,
            self.labels,
        )

    def _handle_result(self, result: PartialResultMessage[PatchResult]):
        # The workers save the files,
        # so no need to do anything except log results
        print(
            f"Received patch {result.data.patch_label}.",
            f"Loss: {result.data.val_loss:.3f}.",
            f"Success rate: {result.data.percent_successful:.3f}.",
        )

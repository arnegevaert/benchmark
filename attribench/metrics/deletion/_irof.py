from typing import Dict, Callable, Union, Tuple, List
import torch
import numpy as np
from torch import nn
from torch import multiprocessing as mp
from torch.utils.data import DataLoader

from attribench.masking import ImageMasker
from attribench.distributed import (
    Worker,
    DistributedSampler,
    PartialResultMessage,
)
from attribench.result._batch_result import BatchResult
from ._deletion import Deletion
from attribench.data import AttributionsDataset

from ._deletion import DeletionWorker
from ._dataset import _IrofDataset
from ._get_predictions import _get_predictions


def irof(
    samples: torch.Tensor,
    labels: torch.Tensor,
    model: Callable,
    attrs: np.ndarray,
    masker: ImageMasker,
    activation_fns: Union[List[str], str] = "linear",
    mode: str = "morf",
    start: float = 0.0,
    stop: float = 1.0,
    num_steps: int = 100,
):
    masking_dataset = _IrofDataset(
        mode, start, stop, num_steps, samples, masker
    )
    if type(activation_fns) == str:
        activation_fns = [activation_fns]
    masking_dataset.set_attrs(attrs)
    return _get_predictions(masking_dataset, labels, model, activation_fns)


class IrofWorker(DeletionWorker):
    def work(self):
        sampler = DistributedSampler(
            self.dataset, self.world_size, self.rank, shuffle=False
        )
        dataloader = DataLoader(
            self.dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=4,
        )
        device = torch.device(self.rank)
        model = self.model_factory()
        model.to(device)

        for (
            batch_indices,
            batch_x,
            batch_y,
            batch_attr,
            method_names,
        ) in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                if not isinstance(masker, ImageMasker):
                    raise ValueError("Invalid masker", masker_name)
                batch_result[masker_name] = irof(
                    batch_x,
                    batch_y,
                    model,
                    batch_attr.numpy(),
                    masker,
                    self.activation_fns,
                    self.mode,
                    self.start,
                    self.stop,
                    self.num_steps,
                )
            self.send_result(
                PartialResultMessage(
                    self.rank,
                    BatchResult(batch_indices, batch_result, method_names),
                )
            )


class Irof(Deletion):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, ImageMasker],
        activation_fns: Union[Tuple[str], str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Tuple = None,
    ):
        super().__init__(
            model_factory,
            dataset,
            batch_size,
            maskers,
            activation_fns,
            mode,
            start,
            stop,
            num_steps,
            address,
            port,
            devices,
        )

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> Worker:
        return IrofWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.maskers,
            self.activation_fns,
            self.mode,
            self._start,
            self.stop,
            self.num_steps,
            self._handle_result if self.world_size == 1 else None,
        )

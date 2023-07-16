from numpy import typing as npt
import torch
from .._metric_worker import MetricWorker
from ..._worker import WorkerConfig
from typing import Callable, Dict

from torch import nn

from attribench.masking import Masker
from attribench.data import AttributionsDataset
from attribench.functional.metrics.minimal_subset._minimal_subset import (
    minimal_subset_batch,
)


class MinimalSubsetWorker(MetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        mode: str,
        num_steps: int,
    ):
        super().__init__(worker_config, model_factory, dataset, batch_size)
        self.maskers = maskers
        self.mode = mode
        self.num_steps = num_steps

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: torch.Tensor,
    ):
        return minimal_subset_batch(
            batch_x,
            self.model,
            batch_attr,
            self.num_steps,
            self.maskers,
            self.mode,
        )

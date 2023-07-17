import torch
from .._metric_worker import MetricWorker, WorkerConfig
from typing import Mapping, List
from attribench.masking import Masker
from attribench.data import AttributionsDataset
from attribench.functional.metrics.deletion._deletion import _deletion_batch
from attribench._model_factory import ModelFactory


class DeletionWorker(MetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Mapping[str, Masker],
        activation_fns: List[str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
    ):
        super().__init__(worker_config, model_factory, dataset, batch_size)
        self.maskers = maskers
        self.activation_fns = activation_fns
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: torch.Tensor,
    ):
        return _deletion_batch(
            batch_x,
            batch_y,
            self.model,
            batch_attr,
            self.maskers,
            self.activation_fns,
            self.mode,
            self.start,
            self.stop,
            self.num_steps,
        )
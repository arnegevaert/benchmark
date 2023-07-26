from typing import List, Mapping
import torch
from attribench.masking.image import ImageMasker
from attribench._model_factory import ModelFactory
from ..deletion._deletion_worker import DeletionWorker
from .._metric_worker import WorkerConfig
from attribench.data import AttributionsDataset
from attribench.functional.metrics._irof import _irof_batch


class IrofWorker(DeletionWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Mapping[str, ImageMasker],
        activation_fns: List[str],
        mode: str = "morf",
        start: float = 0,
        stop: float = 1,
        num_steps: int = 100,
    ):
        super().__init__(
            worker_config,
            model_factory,
            dataset,
            batch_size,
            maskers,
            activation_fns,
            mode,
            start,
            stop,
            num_steps,
        )
        self.maskers = maskers

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: torch.Tensor,
    ):
        return _irof_batch(
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

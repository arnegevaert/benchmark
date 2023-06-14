from typing import Callable, Tuple, Dict

from torch import nn
from torch import multiprocessing as mp

from attribench.masking import Masker
from attribench.metrics.minimal_subset import (
    MinimalSubsetResult,
    MinimalSubsetWorker,
)
from attribench.metrics import DistributedMetric
from attribench.data import AttributionsDataset


class MinimalSubset(DistributedMetric):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        mode: str = "deletion",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Tuple = None,
    ):
        super().__init__(
            model_factory, dataset, batch_size, address, port, devices
        )
        self.num_steps = num_steps
        self.stop = stop
        self._start = start
        if mode not in ["deletion", "insertion"]:
            raise ValueError("Mode must be deletion or insertion. Got:", mode)
        self.mode = mode
        self.maskers = maskers
        self._result = MinimalSubsetResult(
            dataset.method_names,
            tuple(maskers.keys()),
            mode,
            shape=(dataset.num_samples, 1),
        )

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> MinimalSubsetWorker:
        return MinimalSubsetWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.dataset,
            self.batch_size,
            self.maskers,
            self.mode,
            self.num_steps,
            self._handle_result if self.world_size == 1 else None,
        )

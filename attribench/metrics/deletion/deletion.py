from typing import Callable, Union, Tuple, Dict, Optional

from torch import nn
from torch import multiprocessing as mp

from attribench.masking import Masker
from attribench.metrics.deletion import (
    DeletionResult,
    DeletionWorker,
    InsertionResult,
)
from attribench.metrics import DistributedMetric
from attribench.data import AttributionsDataset


class Deletion(DistributedMetric):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        activation_fns: Union[Tuple[str], str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        super().__init__(
            model_factory, dataset, batch_size, address, port, devices
        )
        self.num_steps = num_steps
        self.stop = stop
        self._start = start
        self.mode = mode
        if isinstance(activation_fns, str):
            activation_fns = (activation_fns,)
        self.activation_fns = activation_fns
        self.maskers = maskers
        self._result = DeletionResult(
            dataset.method_names,
            tuple(maskers.keys()),
            self.activation_fns,
            mode,
            shape=(dataset.num_samples, num_steps),
        )
        self.dataset = dataset

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> DeletionWorker:
        return DeletionWorker(
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


class Insertion(Deletion):
    """
    Computes the insertion metric for a given model and dataset.
    This is a simple wrapper around the deletion metric.
    The only differences are:
    - Start, stop and mode are swapped (deleting the first x% of the input is
      the same as inserting the last x% of the input)
    - The result is an InsertionResult instead of a DeletionResult
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        activation_fns: Union[Tuple[str], str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        super().__init__(
            model_factory,
            dataset,
            batch_size,
            maskers,
            activation_fns,
            "lerf" if mode == "morf" else "morf",  # Swap mode
            1 - start,  # Swap start
            1 - stop,  # Swap stop
            num_steps,
            address,
            port,
            devices,
        )
        self._result = InsertionResult(
            dataset.method_names,
            tuple(maskers.keys()),
            self.activation_fns,
            mode,
            shape=(dataset.num_samples, num_steps),
        )

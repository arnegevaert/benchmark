from typing import Callable, Union, Tuple, Dict

from torch import nn
from torch import multiprocessing as mp

from attrbench.masking import Masker
from attrbench.distributed import Worker
from attrbench.distributed.metrics.deletion import DeletionResult, DeletionWorker
from attrbench.distributed.metrics import DistributedMetric
from attrbench.data import AttributionsDataset


class DistributedDeletion(DistributedMetric):
    def __init__(self, model_factory: Callable[[], nn.Module],
                 dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], activation_fns: Union[Tuple[str], str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100,
                 address="localhost", port="12355", devices: Tuple = None):
        super().__init__(model_factory, dataset, batch_size, address, port, devices)
        self.num_steps = num_steps
        self.stop = stop
        self._start = start
        self.mode = mode
        self.activation_fns = [activation_fns] if isinstance(activation_fns, str) else list(activation_fns)
        self.maskers = maskers
        self._result = DeletionResult(dataset.method_names, list(maskers.keys()),
                                      self.activation_fns, mode, shape=(dataset.num_samples, num_steps))

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> Worker:
        return DeletionWorker(queue, rank, self.world_size, all_processes_done, self.model_factory, self.dataset,
                              self.batch_size, self.maskers, self.activation_fns, self.mode,
                              self._start, self.stop, self.num_steps)

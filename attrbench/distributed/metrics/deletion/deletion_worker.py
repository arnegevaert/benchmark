from attrbench.distributed.metrics import MetricWorker
from typing import Callable, Dict, Tuple

from torch import nn
from torch import multiprocessing as mp

from attrbench.masking import Masker
from attrbench.distributed import DoneMessage, PartialResultMessage
from attrbench.distributed.metrics.result import BatchResult
from attrbench.metrics.deletion import deletion
from attrbench.data import AttributionsDataset


class DeletionWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], activation_fns: Tuple[str], mode: str = "morf",
                 start: float = 0., stop: float = 1., num_steps: int = 100):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.maskers = maskers
        self.activation_fns = activation_fns
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr, method_names in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                batch_result[masker_name] = deletion(batch_x, batch_y, model, batch_attr.numpy(), masker,
                                                     self.activation_fns, self.mode, self.start, self.stop,
                                                     self.num_steps)
            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))

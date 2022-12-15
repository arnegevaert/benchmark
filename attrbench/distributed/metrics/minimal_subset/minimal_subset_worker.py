from attrbench.distributed.metrics import MetricWorker
from typing import Callable, Dict

from torch import nn
from torch import multiprocessing as mp

from attrbench.masking import Masker
from attrbench.distributed import DoneMessage, PartialResultMessage
from attrbench.distributed.metrics.result import BatchResult
from attrbench.metrics.minimal_subset import minimal_subset_deletion, minimal_subset_insertion
from attrbench.data import AttributionsDataset


class MinimalSubsetWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], mode: str, num_steps: int):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.maskers = maskers
        self.mode = mode
        self.num_steps = num_steps
        self.metric_fn = minimal_subset_deletion if mode == "deletion" else minimal_subset_insertion

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr, method_names in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                batch_result[masker_name] = self.metric_fn(batch_x, model, batch_attr.detach().cpu().numpy(),
                                                           self.num_steps, masker)
            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))


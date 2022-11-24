from torch import multiprocessing as mp
from typing import Callable, Dict, Tuple
from torch import nn
from attrbench.data import AttributionsDataset
from attrbench.masking import Masker
from attrbench.distributed.metrics import MetricWorker
from attrbench.distributed.metrics.result import BatchResult
from attrbench.distributed import PartialResultMessage, DoneMessage
from attrbench.metrics import sensitivity_n


class SensitivityNWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                 maskers: Dict[str, Masker], activation_fns: Tuple[str]):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.activation_fns = activation_fns
        self.maskers = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr, method_names in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                batch_result[masker_name] = sensitivity_n(batch_x, batch_y, model, batch_attr.numpy(),
                                                          self.min_subset_size, self.max_subset_size, self.num_steps,
                                                          self.num_subsets, masker, self.activation_fns)
            print(method_names)
            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))
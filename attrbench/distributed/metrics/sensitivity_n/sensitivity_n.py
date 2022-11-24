from attrbench.distributed.metrics import DistributedMetric
from torch import nn
from typing import Callable, Dict, Union, Tuple
from attrbench.data import AttributionsDataset
from attrbench.masking import Masker
from attrbench.distributed.metrics.sensitivity_n import SensitivityNResult, SensitivityNWorker
from torch import multiprocessing as mp


class DistributedSensitivityN(DistributedMetric):
    def __init__(self, model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                 maskers: Dict[str, Masker], activation_fns: Union[Tuple[str], str]):
        super().__init__(model_factory, dataset, batch_size)
        self.activation_fns = [activation_fns] if isinstance(activation_fns, str) else list(activation_fns)
        self.maskers = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size
        self._result = SensitivityNResult(dataset.method_names, list(maskers.keys()),
                                          self.activation_fns, shape=(dataset.num_samples, num_steps))

    def _create_worker(self, queue: mp.Queue, rank: int, all_processes_done: mp.Event) -> SensitivityNWorker:
        return SensitivityNWorker(queue, rank, self.world_size, all_processes_done, self.model_factory,
                                  self.dataset, self.batch_size, self.min_subset_size, self.max_subset_size,
                                  self.num_steps, self.num_subsets, self.maskers, self.activation_fns)

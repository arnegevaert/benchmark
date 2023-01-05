from attrbench.distributed.metrics.distributed_metric import \
        DistributedMetric, MetricWorker
import torch
from typing import Callable, Tuple, Optional, NewType, Dict
from torch import nn
from torch import multiprocessing as mp
from torch.utils.data import Dataset

from .impact_coverage_worker import ImpactCoverageWorker


AttributionMethod = NewType("AttributionMethod",
                            Callable[[torch.Tensor, torch.Tensor],
                                     torch.Tensor])


class DistributedImpactCoverage(DistributedMetric):
    def __init__(self, model_factory: Callable[[], nn.Module],
                 dataset: Dataset, batch_size: int,
                 method_factory: Callable[[nn.Module], 
                                          Dict[str, AttributionMethod]],
                 patch_folder: str,
                 address="localhost",
                 port="12355", devices: Optional[Tuple] = None):
        super().__init__(model_factory, dataset, batch_size, address, port, 
                         devices)
        self.method_factory = method_factory
        self.patch_folder = patch_folder

    def _create_worker(self, queue: mp.Queue, rank: int, 
                       all_processes_done: mp.Event) -> MetricWorker:
        return ImpactCoverageWorker(queue, rank, self.world_size, 
                                    all_processes_done, self.model_factory,
                                    self.method_factory, self.dataset,
                                    self.batch_size, self.patch_folder)


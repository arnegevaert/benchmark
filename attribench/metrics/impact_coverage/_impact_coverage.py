from attribench.metrics._metric import Metric
from attribench.metrics._metric_worker import MetricWorker
from attribench.result import ImpactCoverageResult

from typing import Callable, Tuple, Optional
from torch import nn
from torch import multiprocessing as mp
from torch.utils.data import Dataset
from attribench.data import IndexDataset
from attribench._method_factory import MethodFactory

from ._impact_coverage_worker import ImpactCoverageWorker


class ImpactCoverage(Metric):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: Dataset,
        batch_size: int,
        method_factory: MethodFactory,
        patch_folder: str,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        super().__init__(
            model_factory,
            IndexDataset(dataset),
            batch_size,
            address,
            port,
            devices,
        )
        self.method_factory = method_factory
        self.patch_folder = patch_folder
        self._result = ImpactCoverageResult(
            method_factory.get_method_names(), shape=(len(dataset),)
        )

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> MetricWorker:
        return ImpactCoverageWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.method_factory,
            self.dataset,
            self.batch_size,
            self.patch_folder,
            self._handle_result if self.world_size == 1 else None,
        )

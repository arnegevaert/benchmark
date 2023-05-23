from attrbench.data.index_dataset import IndexDataset
from torch import multiprocessing as mp
from attrbench.metrics import MetricWorker
from typing import Callable, Optional, Tuple
from torch import nn
from attrbench.metrics.distributed_metric import DistributedMetric
from attrbench.method_factory import MethodFactory
from .result import MaxSensitivityResult
from .max_sensitivity_worker import MaxSensitivityWorker


class MaxSensitivity(DistributedMetric):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: IndexDataset,
        batch_size: int,
        method_factory: MethodFactory,
        num_perturbations: int,
        radius: float,
        address="localhost",
        port="12355",
        devices: Optional[Tuple] = None,
    ):
        super().__init__(
            model_factory, dataset, batch_size, address, port, devices
        )
        self.method_factory = method_factory
        self.num_perturbations = num_perturbations
        self.radius = radius
        self._result = MaxSensitivityResult(
            method_factory.get_method_names(), shape=(len(dataset),)
        )

    def _create_worker(
        self, queue: mp.Queue, rank: int, all_processes_done: mp.Event
    ) -> MetricWorker:
        return MaxSensitivityWorker(
            queue,
            rank,
            self.world_size,
            all_processes_done,
            self.model_factory,
            self.method_factory,
            self.dataset,
            self.batch_size,
            self.num_perturbations,
            self.radius,
            self._handle_result if self.world_size == 1 else None
        )

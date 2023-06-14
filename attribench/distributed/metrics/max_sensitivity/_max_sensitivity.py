from attribench.data._index_dataset import IndexDataset
from torch import multiprocessing as mp
from .._metric_worker import MetricWorker
from .._metric import Metric
from typing import Callable, Optional, Tuple
from torch import nn
from torch.utils.data import Dataset
from attribench import MethodFactory
from attribench.result._max_sensitivity_result import MaxSensitivityResult
from ._max_sensitivity_worker import MaxSensitivityWorker


class MaxSensitivity(Metric):
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        dataset: Dataset,
        batch_size: int,
        method_factory: MethodFactory,
        num_perturbations: int,
        radius: float,
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
            self._handle_result if self.world_size == 1 else None,
        )

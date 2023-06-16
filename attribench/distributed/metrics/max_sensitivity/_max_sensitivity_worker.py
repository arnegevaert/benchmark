from attribench.data import IndexDataset
from ..._message import PartialResultMessage
from .._metric_worker import MetricWorker
from typing import Callable, Optional
from torch import nn
from torch import multiprocessing as mp
from attribench.result._batch_result import BatchResult
from attribench.functional.metrics._max_sensitivity import (
    max_sensitivity_batch,
)

from attribench._method_factory import MethodFactory


class MaxSensitivityWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done,
        model_factory: Callable[[], nn.Module],
        method_factory: MethodFactory,
        dataset: IndexDataset,
        batch_size: int,
        num_perturbations: int,
        radius: float,
        result_handler: Optional[
            Callable[[PartialResultMessage], None]
        ] = None,
    ):
        super().__init__(
            result_queue,
            rank,
            world_size,
            all_processes_done,
            model_factory,
            dataset,
            batch_size,
            result_handler,
        )
        self.method_factory = method_factory
        self.num_perturbations = num_perturbations
        self.radius = radius

    def work(self):
        model = self._get_model()

        # Get method dictionary
        method_dict = self.method_factory(model)

        for batch_indices, batch_x, batch_y in self.dataloader:
            batch_result = max_sensitivity_batch(
                batch_x,
                batch_y,
                method_dict,
                self.num_perturbations,
                self.radius,
                self.device,
            )
            self.send_result(
                PartialResultMessage(
                    self.rank, BatchResult(batch_indices, batch_result)
                )
            )

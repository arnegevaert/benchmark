from attribench.data import IndexDataset
from itertools import cycle
import os
from ..._message import PartialResultMessage
from .._metric_worker import MetricWorker, WorkerConfig
from typing import Callable
from torch import nn

from attribench.result._grouped_batch_result import GroupedBatchResult
from attribench._method_factory import MethodFactory
from attribench.functional.metrics._impact_coverage import (
    _impact_coverage_batch,
)


class ImpactCoverageWorker(MetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: Callable[[], nn.Module],
        dataset: IndexDataset,
        batch_size: int,
        method_factory: MethodFactory,
        patch_folder: str,
    ):
        super().__init__(
            worker_config,
            model_factory,
            dataset,
            batch_size,
        )
        self.patch_folder = patch_folder
        self.method_factory = method_factory
        patch_names = [
            filename
            for filename in os.listdir(self.patch_folder)
            if filename.endswith(".pt")
        ]
        self.patch_names_cycle = cycle(patch_names)

    def setup(self):
        self.model = self._get_model()
        self.method_dict = self.method_factory(self.model)

    def work(self):
        self.setup()

        for batch_indices, batch_x, batch_y in self.dataloader:
            # Compute batch result
            batch_result = _impact_coverage_batch(
                self.model,
                self.method_dict,
                batch_x,
                batch_y,
                self.patch_folder,
                self.patch_names_cycle,
                self.device,
            )
            # Return batch result
            self.worker_config.send_result(
                PartialResultMessage(
                    self.worker_config.rank,
                    GroupedBatchResult(batch_indices, batch_result),
                )
            )

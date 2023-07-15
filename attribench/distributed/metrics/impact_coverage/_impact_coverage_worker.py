from attribench.data import IndexDataset
import re
from itertools import cycle
import os
from ..._message import PartialResultMessage
from .._metric_worker import MetricWorker
from torch import multiprocessing as mp
from typing import Callable, Optional
from torch import nn

from attribench.result._grouped_batch_result import GroupedBatchResult
from attribench._method_factory import MethodFactory
from attribench.functional.metrics._impact_coverage import impact_coverage_batch


class ImpactCoverageWorker(MetricWorker):
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
        patch_folder: str,
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
        self.patch_folder = patch_folder
        self.method_factory = method_factory
        patch_names = [
            filename
            for filename in os.listdir(self.patch_folder)
            if filename.endswith(".pt")
        ]
        self.patch_names_cycle = cycle(patch_names)

    def work(self):
        model = self._get_model()

        # Get method dictionary
        method_dict = self.method_factory(model)

        for batch_indices, batch_x, batch_y in self.dataloader:
            # Compute batch result
            batch_result = impact_coverage_batch(
                model,
                method_dict,
                batch_x,
                batch_y,
                self.patch_folder,
                self.patch_names_cycle,
                self.device,
            )
            # Return batch result
            self.send_result(
                PartialResultMessage(
                    self.rank, GroupedBatchResult(batch_indices, batch_result)
                )
            )

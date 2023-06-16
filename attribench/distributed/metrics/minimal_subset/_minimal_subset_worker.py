from .._metric_worker import MetricWorker
from typing import Callable, Dict, Optional

from torch import nn
from torch import multiprocessing as mp

from attribench.masking import Masker
from ..._message import PartialResultMessage
from attribench.result._batch_result import BatchResult
from attribench.data import AttributionsDataset
from attribench.functional.metrics.minimal_subset._minimal_subset import (
    minimal_subset_batch,
)

from multiprocessing.synchronize import Event


class MinimalSubsetWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: Event,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Dict[str, Masker],
        mode: str,
        num_steps: int,
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
        self.maskers = maskers
        self.mode = mode
        self.num_steps = num_steps

    def work(self):
        model = self._get_model()

        for (
            batch_indices,
            batch_x,
            _,
            batch_attr,
            method_names,
        ) in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_result = minimal_subset_batch(
                batch_x,
                model,
                batch_attr,
                self.num_steps,
                self.maskers,
                self.mode,
            )
            self.send_result(
                PartialResultMessage(
                    self.rank,
                    BatchResult(batch_indices, batch_result, method_names),
                )
            )

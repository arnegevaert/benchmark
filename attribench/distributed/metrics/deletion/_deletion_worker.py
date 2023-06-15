from .._metric_worker import MetricWorker
from typing import Callable, Mapping, List, Optional

from torch import multiprocessing as mp
from multiprocessing.synchronize import Event

from attribench.masking import Masker
from ..._message import PartialResultMessage
from attribench.result._batch_result import BatchResult
from attribench.data import AttributionsDataset
from attribench.functional.metrics.deletion._deletion import _deletion_batch
from attribench._model_factory import ModelFactory


class DeletionWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: Event,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        maskers: Mapping[str, Masker],
        activation_fns: List[str],
        mode: str = "morf",
        start: float = 0.0,
        stop: float = 1.0,
        num_steps: int = 100,
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
        self.activation_fns = activation_fns
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps

    def work(self):
        model = self._get_model()

        for (
            batch_indices,
            batch_x,
            batch_y,
            batch_attr,
            method_names,
        ) in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_result = _deletion_batch(
                batch_x,
                batch_y,
                model,
                batch_attr.numpy(),
                self.maskers,
                self.activation_fns,
                self.mode,
                self.start,
                self.stop,
                self.num_steps,
            )
            self.send_result(
                PartialResultMessage(
                    self.rank,
                    BatchResult(batch_indices, batch_result, method_names),
                )
            )

from torch import multiprocessing as mp
import numpy as np
from typing import Callable, Dict, Optional, List
from attribench.data import AttributionsDataset
from attribench.masking import Masker
from .._metric_worker import MetricWorker
from attribench.result._batch_result import BatchResult
from ..._message import PartialResultMessage
from attribench._model_factory import ModelFactory
from attribench.functional.metrics.sensitivity_n._sensitivity_n import _sens_n_batch
from multiprocessing.synchronize import Event


class SensitivityNWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: Event,
        model_factory: ModelFactory,
        dataset: AttributionsDataset,
        batch_size: int,
        min_subset_size: float,
        max_subset_size: float,
        num_steps: int,
        num_subsets: int,
        maskers: Dict[str, Masker],
        activation_fns: List[str],
        segmented=False,
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
        self.dataset = dataset
        self.activation_fns = activation_fns
        self.maskers: Dict[str, Masker] = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size
        self.segmented = segmented
        self.n_range = np.linspace(
            self.min_subset_size, self.max_subset_size, self.num_steps
        )

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            batch_result = _sens_n_batch(
                batch_x,
                batch_y,
                model,
                batch_attr,
                self.maskers,
                self.activation_fns,
                self.n_range,
                self.num_subsets,
                self.segmented,
            )

            self.send_result(
                PartialResultMessage(
                    self.rank, BatchResult(batch_indices, batch_result)
                )
            )

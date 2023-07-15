from torch import multiprocessing as mp
from typing import Callable, Dict, Tuple, Optional
from torch import nn
from attribench.data import AttributionsDataset
from .._metric_worker import MetricWorker
from attribench.result._grouped_batch_result import GroupedBatchResult
from ..._message import PartialResultMessage
from attribench.functional.metrics.infidelity._perturbation_generator import (
    PerturbationGenerator,
)
from multiprocessing.synchronize import Event
from attribench.functional.metrics.infidelity._infidelity import (
    infidelity_batch,
)


class InfidelityWorker(MetricWorker):
    def __init__(
        self,
        result_queue: mp.Queue,
        rank: int,
        world_size: int,
        all_processes_done: Event,
        model_factory: Callable[[], nn.Module],
        dataset: AttributionsDataset,
        batch_size: int,
        perturbation_generators: Dict[str, PerturbationGenerator],
        num_perturbations: int,
        activation_fns: Tuple[str],
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
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr in self.dataloader:
            batch_result = infidelity_batch(
                model,
                batch_x,
                batch_y,
                batch_attr,
                self.perturbation_generators,
                self.num_perturbations,
                self.activation_fns,
                self.device,
            )
            # Return batch result
            self.send_result(
                PartialResultMessage(
                    self.rank, GroupedBatchResult(batch_indices, batch_result)
                )
            )

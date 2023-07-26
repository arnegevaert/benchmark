import torch
from typing import Callable, Dict, List
from torch import nn
from attribench.data.attributions_dataset._attributions_dataset import GroupedAttributionsDataset
from .._metric_worker import GroupedMetricWorker, WorkerConfig
from attribench.functional.metrics.infidelity._perturbation_generator import (
    PerturbationGenerator,
)
from attribench.functional.metrics.infidelity._infidelity import (
    _infidelity_batch,
)


class InfidelityWorker(GroupedMetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: Callable[[], nn.Module],
        dataset: GroupedAttributionsDataset,
        batch_size: int,
        perturbation_generators: Dict[str, PerturbationGenerator],
        num_perturbations: int,
        activation_fns: List[str],
    ):
        super().__init__(
            worker_config,
            model_factory,
            dataset,
            batch_size,
        )
        self.activation_fns = activation_fns
        self.num_perturbations = num_perturbations
        self.perturbation_generators = perturbation_generators

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: Dict[str, torch.Tensor],
    ):
        return _infidelity_batch(
            self.model,
            batch_x,
            batch_y,
            batch_attr,
            self.perturbation_generators,
            self.num_perturbations,
            self.activation_fns,
            self.device,
        )

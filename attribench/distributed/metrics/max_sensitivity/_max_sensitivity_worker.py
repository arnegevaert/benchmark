import torch
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
)
from .._metric_worker import GroupedMetricWorker, WorkerConfig
from typing import Callable, Dict
from torch import nn
from attribench.functional.metrics._max_sensitivity import (
    _max_sensitivity_batch,
)
from attribench._method_factory import MethodFactory
from attribench._model_factory import ModelFactory


class MaxSensitivityWorker(GroupedMetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: ModelFactory,
        dataset: GroupedAttributionsDataset,
        batch_size: int,
        method_factory: MethodFactory,
        num_perturbations: int,
        radius: float,
    ):
        super().__init__(
            worker_config,
            model_factory,
            dataset,
            batch_size,
        )
        self.method_factory = method_factory
        self.num_perturbations = num_perturbations
        self.radius = radius

    def setup(self):
        self.model = self._get_model()
        self.method_dict = self.method_factory(self.model)

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: Dict[str, torch.Tensor],
    ):
        return _max_sensitivity_batch(
            batch_x,
            batch_y,
            batch_attr,
            self.method_dict,
            self.num_perturbations,
            self.radius,
            self.device,
        )

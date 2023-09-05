from attribench.distributed._worker import WorkerConfig
import torch
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
)
from .._metric_worker import GroupedMetricWorker, WorkerConfig
from typing import Callable, Dict
from torch import nn
from attribench.functional.metrics._parameter_randomization import (
    _parameter_randomization_batch,
)
from attribench._method_factory import MethodFactory
from attribench._model_factory import ModelFactory


class ParameterRandomizationWorker(GroupedMetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: ModelFactory,
        dataset: GroupedAttributionsDataset,
        batch_size: int,
        method_factory: MethodFactory,
        agg_fn: Callable[
            [
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ]
        | None = None,
        agg_dim: int | None = None,
    ):
        super().__init__(worker_config, model_factory, dataset, batch_size)
        self.method_factory = method_factory
        self.agg_fn = agg_fn
        self.agg_dim = agg_dim

    def setup(self):
        self.randomized_model = self._get_model()
        for layer in self.randomized_model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.method_dict_rand = self.method_factory(self.randomized_model)

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: Dict[str, torch.Tensor],
    ):
        return _parameter_randomization_batch(
            batch_x,
            batch_y,
            batch_attr,
            self.method_dict_rand,
            self.device,
            self.agg_fn,
            self.agg_dim,
        )

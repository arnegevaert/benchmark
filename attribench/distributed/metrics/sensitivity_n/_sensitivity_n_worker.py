import numpy as np
from typing import Dict, List

import torch
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
)
from attribench.masking import Masker
from .._metric_worker import GroupedMetricWorker, WorkerConfig
from attribench._model_factory import ModelFactory
from attribench.functional.metrics.sensitivity_n._sensitivity_n import (
    _sens_n_batch,
)


class SensitivityNWorker(GroupedMetricWorker):
    def __init__(
        self,
        worker_config: WorkerConfig,
        model_factory: ModelFactory,
        dataset: GroupedAttributionsDataset,
        batch_size: int,
        min_subset_size: float,
        max_subset_size: float,
        num_steps: int,
        num_subsets: int,
        maskers: Dict[str, Masker],
        activation_fns: List[str],
        segmented=False,
    ):
        super().__init__(worker_config, model_factory, dataset, batch_size)
        self.dataset = dataset
        self.activation_fns = activation_fns
        self.maskers: Dict[str, Masker] = maskers
        self.num_subsets = num_subsets
        self.num_steps = num_steps
        self.max_subset_size = max_subset_size
        self.min_subset_size = min_subset_size
        self.segmented = segmented
        n_range = np.linspace(
            self.min_subset_size, self.max_subset_size, self.num_steps
        )
        if segmented:
            n_range = n_range * 100
        else:
            total_num_features = np.prod(dataset.attributions_shape)
            n_range = n_range * total_num_features
        self.n_range = n_range.astype(int)

    def process_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_attr: Dict[str, torch.Tensor],
    ):
        return _sens_n_batch(
            batch_x,
            batch_y,
            self.model,
            batch_attr,
            self.maskers,
            self.activation_fns,
            self.n_range,
            self.num_subsets,
            self.segmented,
        )

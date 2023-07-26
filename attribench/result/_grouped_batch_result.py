import torch
from typing import Dict


class GroupedBatchResult:
    """
    Represents results from running a metric on a single batch of images.
    The results are grouped, i.e. the metric is computed for all attribution
    methods at a time. This is used for metrics  which have a shared computation
    for all attribution methods to save computation time, e.g. Infidelity.
    Used by GroupedMetricResult to handle results from distributed metrics.
    """
    def __init__(self, indices: torch.Tensor, results: Dict):
        self.results = results
        self.indices = indices

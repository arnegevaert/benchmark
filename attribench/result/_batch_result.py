import torch
from typing import Dict, List


class BatchResult:
    """
    Represents results from running a metric on a single batch of images.
    Used by MetricResult to handle results from distributed metrics.
    """
    def __init__(self, indices: torch.Tensor, results: Dict,
                 method_names: List[str]):
        self.method_names = method_names
        self.results = results
        self.indices = indices

import torch
from typing import Dict, List


class BatchResult:
    """
    Represents results from running a metric on a single batch of images.
    Used by MetricResult to handle results from distributed metrics.
    The method names in method_names should correspond to the keys in the results dict.
    """
    def __init__(self, indices: torch.Tensor, results: Dict, method_names: List[str]):
        self.method_names = method_names
        self.results = results
        self.indices = indices

        if set(method_names) != set(results.keys()):
            raise ValueError("Invalid results: keys for results should correspond to method names")

from attrbench.metrics import AbstractMetricResult
from typing import List, Callable, Optional
import numpy as np


class Metric:
    def __init__(self, model: Callable, method_names: List[str]):
        self.method_names = method_names
        self.model = model
        self.metadata = {}
        self._result: Optional[AbstractMetricResult] = None

    @property
    def result(self) -> AbstractMetricResult:
        if self._result is not None:
            return self._result
        raise NotImplementedError

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        raise NotImplementedError

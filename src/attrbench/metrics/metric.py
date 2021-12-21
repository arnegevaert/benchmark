from attrbench.lib import AttributionWriter
from attrbench.metrics import AbstractMetricResult
from os import path
from typing import List, Callable, Dict, Optional
import numpy as np


class Metric:
    def __init__(self, model: Callable, method_names: List[str], writer_dir: str = None):
        self.method_names = method_names
        self.model = model
        self.metadata = {}
        self.writer_dir = writer_dir
        self._result: Optional[AbstractMetricResult] = None
        self.writers: Optional[Dict[str, AttributionWriter]] = \
            {method_name: AttributionWriter(path.join(self.writer_dir, method_name), method_name)
             for method_name in method_names} if self.writer_dir is not None else None

    def _get_writer(self, method_name):
        if self.writer_dir is not None:
            writer = self.writers[method_name]
            if writer:
                writer.increment_batch()
            return writer
        return None

    @property
    def result(self) -> AbstractMetricResult:
        if self._result is not None:
            return self._result
        raise NotImplementedError

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        raise NotImplementedError

from attrbench.lib import AttributionWriter
from attrbench.metrics import AbstractMetricResult
from os import path
from typing import List, Callable, Dict, Optional


class Metric:
    result: AbstractMetricResult

    def __init__(self, model: Callable, method_names: List[str], writer_dir: str = None):
        self.model = model
        self.metadata = {}
        self.writer_dir = writer_dir
        self.writers: Optional[Dict[str, AttributionWriter]] = \
            {method_name: AttributionWriter(path.join(self.writer_dir, method_name), method_name)
             for method_name in method_names} if self.writer_dir is not None else None

        # This code checks that any subclass of Metric instantiates the result: MetricResult property defined above
        annotations = self.__class__.__dict__.get('__annotations__', {})
        for name, type_ in annotations.items():
            if not hasattr(self, name):
                raise AttributeError(f'required attribute {name} not present '
                                     f'in {self.__class__}')

    def _get_writer(self, method_name):
        if self.writer_dir is not None:
            writer = self.writers[method_name]
            if writer:
                writer.increment_batch()
            return writer
        return None

    def get_result(self) -> AbstractMetricResult:
        return self.result

    def run_batch(self, samples, labels, attrs_dict: dict):
        raise NotImplementedError

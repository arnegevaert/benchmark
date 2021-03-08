import torch
import numpy as np
from attrbench.lib import AttributionWriter
from os import path
from typing import List, Callable


class Metric:
    def __init__(self, model: Callable, method_names: List[str], writer_dir: str = None):
        self.model = model
        self.results = {method_name: [] for method_name in method_names}
        self.metadata = {}
        self.writer_dir = writer_dir
        self.writers = {method_name: AttributionWriter(path.join(self.writer_dir, method_name))
                        for method_name in method_names} if self.writer_dir is not None else None

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in attrs_dict:
            if method_name not in self.results:
                raise ValueError(f"Invalid method name: {method_name}")
            self.results[method_name].append(self._run_single_method(samples, labels, attrs_dict[method_name],
                                                                     writer=self._get_writer(method_name)))

    def _get_writer(self, method_name):
        if self.writer_dir is not None:
            writer = self.writers[method_name]
            if writer:
                writer.set_method_name(method_name)
                writer.increment_batch()
            return writer
        return None

    def get_results(self):
        """
        Returns the complete results for all batches and all methods in a dictionary
        """
        result = {}
        shape = None
        for method_name in self.results:
            result[method_name] = torch.cat(self.results[method_name], dim=0).numpy()
            if shape is None:
                shape = result[method_name].shape
            elif result[method_name].shape != shape:
                raise ValueError(f"Inconsistent shapes for results: "
                                 f"{method_name} had {result[method_name].shape} instead of {shape}")
        return result, shape

    def _run_single_method(self, samples: torch.Tensor, labels: torch.Tensor, attrs: np.ndarray, writer=None):
        raise NotImplementedError

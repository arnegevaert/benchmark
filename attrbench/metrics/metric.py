import torch
import numpy as np
from attrbench.lib import AttributionWriter
from os import path
from typing import List, Callable, Dict, Optional


class Metric:
    def __init__(self, model: Callable, method_names: List[str], writer_dir: str = None):
        self.model = model
        self.results = {method_name: [] for method_name in method_names}
        self.metadata = {}
        self.writer_dir = writer_dir
        self.writers: Optional[Dict[str, AttributionWriter]] = \
            {method_name: AttributionWriter(path.join(self.writer_dir, method_name)) for method_name in method_names} \
            if self.writer_dir is not None else None

    def run_batch(self, samples, labels, attrs_dict: dict):
        raise NotImplementedError

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

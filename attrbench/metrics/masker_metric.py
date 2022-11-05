from attrbench.metrics import Metric
import masking
from typing import Callable, List, Dict
import numpy as np


class MaskerMetric(Metric):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        # Process "maskers" argument: either the keys are Masker objects,
        # or they are dictionaries that need to be parsed.
        self.maskers = {}
        for key, value in maskers.items():
            if type(value) == masking.Masker:
                self.maskers[key] = value  # Object is already a Masker
            else:
                constructor = getattr(masking, value["type"])
                self.maskers[key] =  constructor(**{k: value[k] for k in value if k != "type"})

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        raise NotImplementedError

from attrbench.metrics import Metric
from attrbench.lib import masking
from typing import Callable, List, Dict
import numpy as np


def _parse_masker(d):
    constructor = getattr(masking, d["type"])
    return constructor(**{key: d[key] for key in d if key != "type"})


class MaskerMetric(Metric):
    def __init__(self, model: Callable, method_names: List[str], maskers: Dict,
                 writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        # Process "maskers" argument: either it is a dictionary of Masker objects,
        # or it is a dictionary that needs to be parsed.
        self.maskers = {}
        for key, value in maskers.items():
            if type(value) == masking.Masker:
                self.maskers[key] = value  # Object is already a Masker
            else:
                self.maskers[key] = _parse_masker(value)  # Object needs to be parsed into a Masker

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        raise NotImplementedError

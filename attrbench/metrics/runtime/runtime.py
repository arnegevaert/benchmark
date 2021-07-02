import torch
from attrbench.metrics import Metric, BasicMetricResult
from typing import Callable, Dict
from timeit import Timer
import numpy as np


def runtime(samples: torch.Tensor, labels: torch.Tensor, method: Callable, single_image=False):
    res = []
    if single_image:
        for i in range(samples.shape[0]):
            timer = Timer(lambda: method(samples[i, ...].unsqueeze(0), labels[i].unsqueeze(0)))
            t = timer.timeit(1)
            # If time is very small, re-run to get more accurate measurements
            if t < 0.01:
                t = timer.timeit(number=100) / 100
            res.append(t)
    else:
        timer = Timer(lambda: method(samples, labels))
        t = timer.timeit(1)
        # If time is very small, re-run to get more accurate measurements
        if t < 0.01:
            t = timer.timeit(number=100) / 100
        res = [t / samples.shape[0]] * samples.shape[0]
    return torch.tensor(res).reshape(-1, 1)


class Runtime(Metric):
    def __init__(self, model, methods: Dict[str, Callable], single_image=False):
        super().__init__(model, list(methods.keys()))
        self.methods = methods
        self.single_image = single_image
        self._result: RuntimeResult = RuntimeResult(list(methods.keys()))

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        batch_result = {method_name: runtime(samples, labels, method,
                                             self.single_image).cpu().detach().numpy()
                        for method_name, method in self.methods.items()}
        self.result.append(batch_result)


class RuntimeResult(BasicMetricResult):
    inverted = True

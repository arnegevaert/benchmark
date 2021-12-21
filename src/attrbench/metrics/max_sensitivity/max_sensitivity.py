import math
from typing import Callable, Dict

import numpy as np
import torch

from attrbench.metrics import Metric
from .result import MaxSensitivityResult


def _normalize_attrs(attrs):
    flattened = attrs.flatten(1)
    return flattened / torch.norm(flattened, dim=1, p=math.inf, keepdim=True)


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, attrs: np.ndarray, radius: float,
                    num_perturbations: int, writer=None):
    device = samples.device
    attrs = _normalize_attrs(torch.tensor(attrs).float())  # [batch_size, -1]

    diffs = []
    for n_p in range(num_perturbations):
        # Add uniform noise with infinity norm <= radius
        # torch.rand generates noise between 0 and 1 => This generates noise between -radius and radius
        noise = torch.rand(samples.shape, device=device) * 2 * radius - radius
        noisy_samples = samples + noise
        if writer:
            writer.add_images("perturbations", noise, global_step=n_p)
            writer.add_images("perturbed_samples", noisy_samples, global_step=n_p)
        # Get new attributions from noisy samples
        noisy_attrs = _normalize_attrs(method(noisy_samples, labels).detach())
        # Get relative norm of attribution difference
        # [batch_size]
        diffs.append(torch.norm(noisy_attrs.cpu() - attrs, dim=1).detach())
    # [batch_size, num_perturbations]
    diffs = torch.stack(diffs, 1)
    # [batch_size]
    result = diffs.max(dim=1, keepdim=True)[0].cpu()
    return result


class MaxSensitivity(Metric):
    def __init__(self, model: Callable, methods: Dict[str, Callable], radius: float,
                 num_perturbations: int, writer_dir: str = None):
        super().__init__(model, list(methods.keys()), writer_dir)
        self.methods = methods
        self.radius = radius
        self.num_perturbations = num_perturbations
        self._result: MaxSensitivityResult = MaxSensitivityResult(method_names=list(methods.keys()) + ["_BASELINE"], radius=radius)
        self.rng = np.random.default_rng()

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in self.methods:
            method = self.methods[method_name]
            max_sens = max_sensitivity(samples, labels, method, attrs_dict[method_name], self.radius,
                                       self.num_perturbations, writer=self._get_writer(method_name))
            self.result.append({method_name: max_sens.cpu().detach().numpy()})

        baseline_result = []
        for i in range(baseline_attrs.shape[0]):
            baseline_result.append(
                max_sensitivity(samples, labels, lambda x, _: torch.tensor(self.rng.random(baseline_attrs.shape[1:])) * 2 - 1,
                                baseline_attrs[i, ...], self.radius, self.num_perturbations)
            )
        self.result.append({"_BASELINE": np.stack(baseline_result, axis=1)})

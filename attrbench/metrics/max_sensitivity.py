from typing import Callable, Dict, List
import h5py
import torch
import math
from attrbench.metrics import Metric, MetricResult


def _normalize_attrs(attrs):
    flattened = attrs.flatten(1)
    return flattened / torch.norm(flattened, dim=1, p=math.inf, keepdim=True)


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, attrs: torch.Tensor, radius: float,
                    num_perturbations: int, writer=None):
    device = samples.device
    if attrs is None:
        attrs = method(samples, labels).detach()  # [batch_size, *sample_shape]
    attrs = _normalize_attrs(attrs)  # [batch_size, -1]

    diffs = []
    for n_p in range(num_perturbations):
        # Add uniform noise with infinity norm <= radius
        # torch.rand generates noise between 0 and 1 => This generates noise between -radius and radius
        noise = torch.rand(samples.shape, device=device) * 2 * radius - radius
        noisy_samples = samples + noise
        if writer:
            writer.add_images('perturbations', noise, global_step=n_p)
            writer.add_images('perturbed_samples', noisy_samples, global_step=n_p)
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
        self.result = MaxSensitivityResult(method_names=list(methods.keys()))

    def run_batch(self, samples, labels, attrs_dict: dict):
        """
        Runs the metric for a given batch, for all methods, and saves result internally
        """
        for method_name in self.methods:
            method = self.methods[method_name]
            max_sens = max_sensitivity(samples, labels, method, attrs_dict[method_name], self.radius,
                                       self.num_perturbations, writer=self._get_writer(method_name))
            self.result.append(method_name, max_sens)


class MaxSensitivityResult(MetricResult):
    def __init__(self, method_names: List[str], radius: float):
        super().__init__(method_names)
        self.inverted = True
        self.radius = radius

    def add_to_hdf(self, group: h5py.Group):
        super().add_to_hdf(group)
        group.attrs["radius"] = self.radius

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        result = cls(method_names, radius=group.attrs["radius"])
        result.data = {m_name: [group[m_name]] for m_name in method_names}
        return result

from typing import List, Tuple

import h5py
import numpy as np

from attrbench.metrics import MetricResult, ModeActivationMetricResult


class InfidelityResult(ModeActivationMetricResult):
    inverted = {
        "mse": True,
        "corr": True
    }

    def __init__(self, method_names: List[str], perturbation_mode: str, perturbation_size: float,
                 modes: Tuple[str], activation_fn: Tuple[str]):
        super().__init__(method_names, modes, activation_fn)
        self.perturbation_mode = perturbation_mode
        self.perturbation_size = perturbation_size

    def add_to_hdf(self, group: h5py.Group):
        super().add_to_hdf(group)
        group.attrs["perturbation_mode"] = self.perturbation_mode
        group.attrs["perturbation_size"] = self.perturbation_size

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        mode = tuple(group[method_names[0]].keys())
        activation_fn = tuple(group[method_names[0]][mode[0]].keys())
        result = cls(method_names, group.attrs["perturbation_mode"], group.attrs["perturbation_size"],
                     mode, activation_fn)
        result.data = {
            m_name:
                {mode:
                    {afn: np.array(group[m_name][mode][afn]) for afn in activation_fn}
                 for mode in mode}
            for m_name in method_names
        }
        return result

from typing import List

import h5py
import numpy as np
import torch

from attrbench.metrics import BasicMetricResult


class ImpactScoreResult(BasicMetricResult):
    inverted = False

    def __init__(self, method_names: List[str], strict: bool, tau: float = None):
        super().__init__(method_names)
        self.data = {
            "flipped": {m_name: [] for m_name in self.method_names},
            "totals": {m_name: [] for m_name in self.method_names}
        }
        self.strict = strict
        self.tau = tau

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            flipped = self.data["flipped"][method_name]
            totals = self.data["totals"][method_name]
            if type(flipped) == list:
                flipped = torch.stack(flipped, dim=0).float().numpy()
                totals = torch.tensor(totals).reshape(-1, 1).float().numpy()
            method_group = group.create_group(method_name)
            method_group.create_dataset("flipped", data=flipped)
            method_group.create_dataset("totals", data=totals)
        group.attrs["strict"] = self.strict
        if self.tau is not None:
            group.attrs["tau"] = self.tau

    def append(self, method_name, batch):
        flipped, total = batch
        self.data["flipped"][method_name].append(flipped)
        self.data["totals"][method_name].append(total)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> BasicMetricResult:
        method_names = list(group.keys())
        tau = group.attrs.get("tau", None)
        result = ImpactScoreResult(method_names, group.attrs["strict"], tau)
        for m_name in method_names:
            result.data["flipped"][m_name] = np.array(group[m_name]["flipped"])
            result.data["totals"][m_name] = np.array(group[m_name]["totals"])
        return result

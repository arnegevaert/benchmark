from attrbench.suite import SuiteResult
import numpy as np
from typing import List


class DFExtractor:
    def __init__(self, res_obj: SuiteResult, exclude_methods=None):
        self.res_obj = res_obj
        self.exclude_methods = exclude_methods
        self.dfs = {}

    def add_metric(self, name, metric, mode=None, activation=None, log_transform=False):
        if name in self.dfs:
            raise ValueError(f"Metric {name} is already present")
        kwargs = {}
        if mode is not None:
            kwargs["mode"] = mode
        if activation is not None:
            kwargs["activation"] = activation
        df, inverted = self.res_obj.metric_results[metric].get_df(**kwargs)
        if log_transform:
            df = df.apply(np.log)
        if self.exclude_methods is not None:
            df = df[df.columns.difference(self.exclude_methods)]
        self.dfs[name] = (df, inverted)

    def add_metrics(self, metric_dict, log_transform=False):
        for key, item in metric_dict.items():
            self.add_metric(key, **item, log_transform=log_transform)

    def get_dfs(self):
        return self.dfs

    def add_infidelity(self, mode, activation):
        self.add_metrics({
            f"infid-gauss-{activation}-{mode}": dict(metric="infidelity_gaussian",
                                                     mode=mode, activation=activation),
            f"infid-seg-{activation}-{mode}": dict(metric="infidelity_seg",
                                                   mode=mode, activation=activation)
        }, log_transform=True)

    def compare_maskers(self, maskers: List[str], activation: str):
        for masker in maskers:
            self.add_metric(f"del_flip-{masker}", f"masker_{masker}.deletion_until_flip")
            for metric in ("insertion", "deletion", "irof", "iiof", "seg_sensitivity_n", "sensitivity_n"):
                self.add_metric(f"{metric}-{masker}", f"masker_{masker}.{metric}", activation=activation)

    def compare_activations(self, activations: List[str], masker: str):
        for activation in activations:
            self.add_metric(f"sens_n-{activation}", f"masker_{masker}.sensitivity_n", activation=activation)
            self.add_metric(f"seg_sens_n-{activation}", f"masker_{masker}.seg_sensitivity_n", activation=activation)
            for metric in ("insertion", "deletion", "irof", "iiof"):
                self.add_metric(f"{metric}-{activation}", f"masker_{masker}.{metric}", mode="morf",
                                activation=activation)

from attrbench.suite import SuiteResult
import numpy as np


class DFExtractor:
    def __init__(self, res_obj: SuiteResult):
        self.res_obj = res_obj
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
        self.dfs[name] = (df, inverted)

    def add_metrics(self, metric_dict, log_transform=False):
        for key, item in metric_dict.items():
            self.add_metric(key, **item, log_transform=log_transform)

    def get_dfs(self):
        return self.dfs

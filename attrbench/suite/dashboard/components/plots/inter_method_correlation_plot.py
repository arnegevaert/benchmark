import pandas as pd
from plotly import express as px

from attrbench.suite.dashboard.components import Component


class InterMethodCorrelationPlot(Component):
    def __init__(self, result_obj, method_name):
        # Take the average metric value for each sample and each metric
        # Only for metrics that have per-sample results (shape[0] > 1)
        data = {metric_name: result_obj.data[metric_name][method_name].mean(axis=1)
                for metric_name in result_obj.get_metrics()
                if result_obj.data[metric_name][method_name].shape[0] > 1}
        self.df = pd.concat(data, axis=1)

    def render(self):
        corrs = self.df.corr(method="spearman")
        return px.imshow(corrs, zmin=-1, zmax=1)


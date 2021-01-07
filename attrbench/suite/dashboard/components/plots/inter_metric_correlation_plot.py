import pandas as pd
from plotly import express as px

from attrbench.suite.dashboard.components import Component


class InterMetricCorrelationPlot(Component):
    def __init__(self, result_obj, metric_name):
        # Take the average metric value for each sample and method, for given metric
        # No need to check shape, this plot should only be called for applicable metrics
        data = {method_name: result_obj.data[metric_name][method_name].mean(axis=1)
                for method_name in result_obj.get_methods()}
        self.df = pd.concat(data, axis=1)

    def render(self):
        corrs = self.df.corr(method="spearman")
        return px.imshow(corrs, zmin=-1, zmax=1)


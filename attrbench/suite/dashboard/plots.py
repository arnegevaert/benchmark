import pandas as pd
import numpy as np
from plotly import express as px
from attrbench.suite.dashboard.component import Component


class Lineplot(Component):
    @staticmethod
    def _normalize(data, mode):
        if mode == "decreasing":
            return {
                method_name:
                    (data[method_name].sub(data[method_name].iloc[:, -1], axis=0))
                    .div(data[method_name].iloc[:, 0] - data[method_name].iloc[:, -1], axis=0)
                for method_name in data
            }
        elif mode == "increasing":
            return {
                method_name:
                    (data[method_name].sub(data[method_name].iloc[:, 0], axis=0))
                    .div(data[method_name].iloc[:, -1] - data[method_name].iloc[:, 0], axis=0)
                for method_name in data
            }
        return data

    def __init__(self, result_obj, metric_name):
        super().__init__()
        data = result_obj.data[metric_name]
        x_ticks = result_obj.metadata[metric_name]["col_index"]
        metric_type = result_obj.metadata[metric_name]["type"]
        normalization = {
            "Insertion": "increasing",
            "Deletion": "decreasing",
        }
        method_names = list(data.keys())
        self.df = pd.DataFrame(columns=method_names, index=x_ticks)
        normalized_data = Lineplot._normalize(data, mode=normalization.get(metric_type, None))
        for method_name in method_names:
            self.df[method_name] = np.average(normalized_data[method_name], axis=0)

    def render(self):
        return px.line(self.df)


class Boxplot(Component):
    def __init__(self, result_obj, metric_name):
        super().__init__()
        data = result_obj.data[metric_name]
        self.df = pd.concat(data, axis=1)
        self.df.columns = self.df.columns.get_level_values(0)

    def render(self):
        return px.box(self.df)


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
        return px.imshow(corrs)


class InterMetricCorrelationPlot(Component):
    def __init__(self, result_obj, metric_name):
        # Take the average metric value for each sample and method, for given metric
        # No need to check shape, this plot should only be called for applicable metrics
        data = {method_name: result_obj.data[metric_name][method_name].mean(axis=1)
                for method_name in result_obj.get_methods()}
        self.df = pd.concat(data, axis=1)

    def render(self):
        corrs = self.df.corr(method="spearman")
        return px.imshow(corrs)



class DendrogramPlot(Component):
    def render(self):
        return None


class EffectSizePlot(Component):
    def render(self):
        return None

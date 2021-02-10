import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import CorrelationMap, BarPlot
from attrbench.suite.dashboard.util import krippendorff_alpha


class CorrelationsPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)
        self.rendered_contents = None

    def render(self) -> html.Div:
        if not self.rendered_contents:
            result = []
            # Krippendorff Alpha
            result.append(html.H2("Krippendorff Alpha"))
            names, values = [], []
            metrics = [m for m in self.result_obj.get_metrics() if self.result_obj.metadata[m]["shape"][0] > 1]
            for metric_name in metrics:
                metric_data = self.result_obj.data[metric_name]
                names.append(metric_name)
                data = np.stack(
                    [metric_data[method_name].mean(axis=1).to_numpy()
                     for method_name in self.result_obj.get_methods()],
                    axis=1)
                values.append(krippendorff_alpha(np.argsort(data)))
            result.append(BarPlot(values, names, id="krippendorff-alpha").render())

            # Inter-metric correlation
            result.append(html.H2("Inter-metric correlations"))
            """
            for method_name in self.result_obj.get_methods():
                result.append(html.H3(method_name))
                data = {metric_name: self.result_obj.data[metric_name][method_name].mean(axis=1)
                        for metric_name in self.result_obj.get_metrics()
                        if self.result_obj.data[metric_name][method_name].shape[0] > 1}
                plot = CorrelationMap(pd.concat(data, axis=1), id=f"{method_name}-metric-corr")
                result.append(plot.render())
            """
            data = {metric_name: [] for metric_name in self.result_obj.get_metrics()
                    if self.result_obj.metadata[metric_name]["shape"][0] > 1}
            for method_name in self.result_obj.get_methods():
                for metric_name in self.result_obj.get_metrics():
                    if self.result_obj.metadata[metric_name]["shape"][0] > 1:
                        data[metric_name].append(self.result_obj.data[metric_name][method_name].mean(axis=1))
            for metric_name in data.keys():
                data[metric_name] = pd.concat(data[metric_name], axis=0)
            inter_metric_corr = CorrelationMap(pd.concat(data, axis=1), id=f"metric-corr")
            result.append(dbc.Row([
                dbc.Col(inter_metric_corr.render()),
            ]))

            """
            # Inter-method correlation
            result.append(html.H2("Inter-method correlations"))
            metrics = [m for m in self.result_obj.get_metrics()
                       if self.result_obj.metadata[m]["shape"][0] == self.result_obj.num_samples]
            for metric_name in metrics:
                result.append(html.H3(metric_name))
                data = {method_name: self.result_obj.data[metric_name][method_name].mean(axis=1)
                        for method_name in self.result_obj.get_methods()}
                plot = CorrelationMap(pd.concat(data, axis=1), id=f"{metric_name}-method-corr")
                result.append(plot.render())
            """

            data = {method_name: [] for method_name in self.result_obj.get_methods()}
            for metric_name in self.result_obj.get_metrics():
                if self.result_obj.metadata[metric_name]["shape"][0] > 1:
                    for method_name in self.result_obj.get_methods():
                        data[method_name].append(self.result_obj.data[metric_name][method_name].mean(axis=1))
            for method_name in data.keys():
                data[method_name] = pd.concat(data[method_name], axis=0)
            inter_method_corr = CorrelationMap(pd.concat(data, axis=1), id="method-corr")
            result.append(dbc.Row([
                dbc.Col(inter_method_corr.render())
            ]))

            self.rendered_contents = html.Div(result)
        return self.rendered_contents

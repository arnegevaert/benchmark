from attrbench.suite.dashboard.plots import *
import dash_html_components as html
import dash_core_components as dcc
import numpy as np
from attrbench.suite.dashboard.util import krippendorff_alpha


class OverviewPage(Component):
    def __init__(self, result_obj):
        super().__init__()
        self.result_obj = result_obj
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            for metric_name in self.result_obj.get_metrics():
                result.append(html.H2(metric_name))
                if len(self.result_obj.metadata[metric_name]["shape"]) > 1:
                    plot = Lineplot(self.result_obj, metric_name)
                else:
                    plot = Boxplot(self.result_obj, metric_name)
                result.append(dcc.Graph(
                    id=metric_name,
                    figure=plot.render()
                ))
            self.rendered_contents = result
            return result
        return self.rendered_contents


class CorrelationsPage(Component):
    def __init__(self, result_obj):
        super().__init__()
        self.result_obj = result_obj
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            # Krippendorff Alpha
            result.append(html.H2("Krippendorff Alpha"))
            names, values = [], []
            for metric_name in self.result_obj.get_metrics():
                metric_data = self.result_obj.data[metric_name]
                metric_metadata = self.result_obj.metadata[metric_name]
                if metric_metadata["shape"][0] == self.result_obj.num_samples:
                    names.append(metric_name)
                    if len(metric_metadata["shape"]) > 1:
                        data = np.stack(
                            [metric_data[method_name].mean(axis=1).to_numpy()
                             for method_name in self.result_obj.get_methods()],
                            axis=1)
                        values.append(krippendorff_alpha(np.argsort(data)))
            result.append(dcc.Graph(
                id="krippendorff-alpha",
                figure=BarPlot(values, names).render()
            ))


            # Inter-metric correlation
            result.append(html.H2("Inter-method correlations"))
            for method_name in self.result_obj.get_methods():
                result.append(html.H3(method_name))
                plot = InterMethodCorrelationPlot(self.result_obj, method_name)
                result.append(dcc.Graph(
                    id=f"{method_name}-metric-corr",
                    figure=plot.render()
                ))

            # Inter-method correlation
            result.append(html.H2("Inter-metric correlations"))
            for metric_name in self.result_obj.get_metrics():
                metric_shape = self.result_obj.metadata[metric_name]["shape"]
                if metric_shape[0] == self.result_obj.num_samples:
                    result.append(html.H3(metric_name))
                    plot = InterMetricCorrelationPlot(self.result_obj, metric_name)
                    result.append(dcc.Graph(
                        id=f"{metric_name}-method-corr",
                        figure=plot.render()
                    ))
            self.rendered_contents = result
            return result
        return self.rendered_contents


class ClusteringPage(Component):
    def render(self):
        return html.P("Clustering page")


class SamplesAttributionsPage(Component):
    def render(self):
        return html.P("Samples and attributions page")
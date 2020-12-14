from attrbench.suite.dashboard.component import Component
from attrbench.suite.dashboard.plots import *
import dash_html_components as html
import dash_core_components as dcc


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
                if "col_index" in self.result_obj.metadata[metric_name].keys():
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
                types = ["DeletionUntilFlip", "Insertion", "Deletion",
                         "Infidelity", "MaxSensitivity", "SensitivityN"]
                if self.result_obj.metadata[metric_name]["type"] in types:
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
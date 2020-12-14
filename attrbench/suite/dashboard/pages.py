from attrbench.suite.dashboard.component import Component
from attrbench.suite.dashboard.plots import Lineplot, Boxplot
import dash_html_components as html
import dash_core_components as dcc


class OverviewPage(Component):
    def __init__(self, app, result_obj):
        super().__init__(app)
        self.result_obj = result_obj
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            for metric_name in self.result_obj.get_metrics():
                result.append(html.H2(metric_name))
                if "col_index" in self.result_obj.metadata[metric_name].keys():
                    plot = Lineplot(self.app, self.result_obj, metric_name)
                else:
                    plot = Boxplot(self.app, self.result_obj, metric_name)
                result.append(dcc.Graph(
                    id=metric_name,
                    figure=plot.render()
                ))
            self.rendered_contents = result
            return result
        return self.rendered_contents


class CorrelationsPage(Component):
    def render(self):
        return html.P("Correlations page")


class ClusteringPage(Component):
    def render(self):
        return html.P("Clustering page")


class SamplesAttributionsPage(Component):
    def render(self):
        return html.P("Samples and attributions page")
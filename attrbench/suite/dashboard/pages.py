from attrbench.suite.dashboard.component import Component
from attrbench.suite.dashboard.plots import Lineplot, Boxplot
import dash_html_components as html
import dash_core_components as dcc


class Page(Component):
    def __init__(self, result_obj, app):
        super().__init__(app)
        self.result_obj = result_obj

    def render(self):
        raise NotImplementedError


class OverviewPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj, app)
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            for metric_name in self.result_obj.get_metrics():
                result.append(html.H2(metric_name))
                if "col_index" in self.result_obj.metadata[metric_name].keys():
                    metric_type = self.result_obj.metadata[metric_name]["type"]
                    if metric_type == "Insertion":
                        plot = Lineplot(self.result_obj.data[metric_name],
                                        self.result_obj.metadata[metric_name]["col_index"], self.app,
                                        normalization="increasing")
                    elif metric_type == "Deletion":
                        plot = Lineplot(self.result_obj.data[metric_name],
                                        self.result_obj.metadata[metric_name]["col_index"], self.app,
                                        normalization="decreasing")
                    else:
                        plot = Lineplot(self.result_obj.data[metric_name],
                                        self.result_obj.metadata[metric_name]["col_index"], self.app)
                else:
                    plot = Boxplot(self.result_obj.data[metric_name], self.app)
                result.append(dcc.Graph(
                    id=metric_name,
                    figure=plot.render()
                ))
            self.rendered_contents = result
            return result
        return self.rendered_contents


class CorrelationsPage(Page):
    def render(self):
        return html.P("Correlations page")


class ClusteringPage(Page):
    def render(self):
        return html.P("Clustering page")

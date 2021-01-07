import dash_core_components as dcc
import dash_html_components as html

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import Lineplot, Boxplot


class OverviewPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            for metric_name in self.result_obj.get_metrics():
                result.append(html.H2(metric_name))
                if self.result_obj.metadata[metric_name]["shape"][1] > 1:
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


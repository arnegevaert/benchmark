import dash_html_components as html
import dash_bootstrap_components as dbc

from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import Lineplot, Boxplot


class OverviewPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)
        self.rendered_contents = None

    def render(self) -> html.Div:
        if not self.rendered_contents:
            result = []
            for metric_name in self.result_obj.get_metrics():
                result.append(html.H2(metric_name))
                if self.result_obj.metadata[metric_name]["shape"][1] > 1:
                    result.append(Lineplot(self.result_obj, metric_name, id=metric_name).render())
                else:
                    result.append(Boxplot(self.result_obj, metric_name, id=metric_name).render())
            self.rendered_contents = html.Div(result)
        return self.rendered_contents


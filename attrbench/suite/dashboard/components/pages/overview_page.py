import dash_html_components as html
import pandas as pd

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
                    x_ticks = self.result_obj.metadata[metric_name]["col_index"]
                    result.append(Lineplot(self.result_obj.data[metric_name], x_ticks, id=metric_name).render())
                else:
                    df = pd.concat(self.result_obj.data[metric_name], axis=1)
                    df.columns = df.columns.get_level_values(0)
                    result.append(Boxplot(df, id=metric_name).render())
            self.rendered_contents = html.Div(result)
        return self.rendered_contents


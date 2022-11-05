from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import ClusterMap
import dash_html_components as html
import pandas as pd


class ClusteringPage(Page):
    def render(self) -> html.Div:
        data = {metric_name: {} for metric_name in self.result_obj.get_metrics()}
        for metric_name in self.result_obj.get_metrics():
            for method_name in self.result_obj.get_methods():
                data[metric_name][method_name] = self.result_obj.data[metric_name][method_name].stack().mean()
        df = pd.DataFrame(data)
        return ClusterMap(df).render()

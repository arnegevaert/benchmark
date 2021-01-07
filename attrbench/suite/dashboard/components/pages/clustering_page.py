from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import GeneralClusterMapPlot
import dash_html_components as html


class ClusteringPage(Page):
    def render(self) -> html.Div:
        return GeneralClusterMapPlot(self.result_obj, aggregate=True).render()

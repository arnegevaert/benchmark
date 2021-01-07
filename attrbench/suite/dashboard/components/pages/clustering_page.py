from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import GeneralClusterMapPlot


class ClusteringPage(Page):
    def render(self):
        plot = GeneralClusterMapPlot(self.result_obj, aggregate=True)
        return plot.render()

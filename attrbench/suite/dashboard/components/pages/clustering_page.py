from attrbench.suite.dashboard import util
from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import cluster_map
import dash_html_components as html
import pandas as pd


class ClusteringPage(Page):
    def render(self) -> html.Div:
        dfs = util.get_dfs(self.result_obj, mode='raw', masker='constant',
                           activation='linear')
        return cluster_map(dfs)

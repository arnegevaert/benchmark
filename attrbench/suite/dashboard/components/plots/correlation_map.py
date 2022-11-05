import dash_html_components as html
from plotly import express as px
import dash_core_components as dcc
from attrbench.suite.dashboard.components import Component


class CorrelationMap(Component):
    def __init__(self, df, id):
        self.df = df
        self.id = id

    def render(self) -> html.Div:
        corrs = self.df.corr(method="spearman")
        return html.Div(dcc.Graph(id=self.id,
                                  figure=px.imshow(corrs, zmin=-1, zmax=1,
                                                   height=40*corrs.shape[0],
                                                   width=40*corrs.shape[0])))

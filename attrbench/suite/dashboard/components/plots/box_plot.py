from plotly import express as px
import dash_core_components as dcc
import dash_html_components as html

from attrbench.suite.dashboard.components import Component


class Boxplot(Component):
    def __init__(self, df, id):
        super().__init__()
        self.df = df
        self.id = id

    def render(self) -> html.Div:
        return html.Div(
            dcc.Graph(
                id=self.id,
                figure=px.box(self.df)
            )
        )

import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc

from attrbench.suite.dashboard.components import Component


class BarPlot(Component):
    def __init__(self, values, names, id):
        self.values = values
        self.names = names
        self.id = id

    def render(self) -> html.Div:
        return html.Div(
            dcc.Graph(
                id=self.id,
                figure=go.Figure([go.Bar(x=self.names, y=self.values)])
            )
        )


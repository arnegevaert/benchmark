import plotly.graph_objects as go

from attrbench.suite.dashboard.components import Component


class BarPlot(Component):
    def __init__(self, values, names):
        self.values = values
        self.names = names

    def render(self):
        return go.Figure([go.Bar(x=self.names, y=self.values)])


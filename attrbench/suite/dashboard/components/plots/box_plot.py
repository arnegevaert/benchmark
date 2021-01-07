import pandas as pd
from plotly import express as px
import dash_core_components as dcc
import dash_html_components as html

from attrbench.suite.dashboard.components import Component


class Boxplot(Component):
    def __init__(self, result_obj, metric_name, id):
        super().__init__()
        data = result_obj.data[metric_name]
        self.df = pd.concat(data, axis=1)
        self.df.columns = self.df.columns.get_level_values(0)
        self.id = id

    def render(self) -> html.Div:
        return html.Div(
            dcc.Graph(
                id=self.id,
                figure=px.box(self.df)
            )
        )

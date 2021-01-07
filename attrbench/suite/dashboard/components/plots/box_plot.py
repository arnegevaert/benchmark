import pandas as pd
from plotly import express as px

from attrbench.suite.dashboard.components import Component


class Boxplot(Component):
    def __init__(self, result_obj, metric_name):
        super().__init__()
        data = result_obj.data[metric_name]
        self.df = pd.concat(data, axis=1)
        self.df.columns = self.df.columns.get_level_values(0)

    def render(self):
        return px.box(self.df)


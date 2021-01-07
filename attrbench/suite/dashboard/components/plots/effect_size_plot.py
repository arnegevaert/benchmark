import numpy as np
import plotly.graph_objects as go

from attrbench.suite.dashboard.components import Component


class EffectSizePlot(Component):
    def __init__(self, df, baseline_name):
        self.df = df
        self.baseline_name = baseline_name
        self.baseline_data = df[baseline_name].to_numpy()

    def render(self):
        x, y, error_x = [], [], []
        for col in self.df.columns:
            if col != self.baseline_name:
                col_data = self.df[col].to_numpy() - self.baseline_data
                x.append(col_data.mean())
                y.append(col)
                error_x.append(1.96 * col_data.std() / np.sqrt(len(col_data)))
        return go.Figure(
            go.Bar(x=x, y=y,
                   error_x=dict(type="data", array=error_x, visible=True),
                   orientation="h"))

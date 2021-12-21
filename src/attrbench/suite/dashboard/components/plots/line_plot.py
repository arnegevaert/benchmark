import numpy as np
import plotly.graph_objects as go
from plotly import express as px
import dash_html_components as html
import dash_core_components as dcc
from itertools import cycle

from attrbench.suite.dashboard.components import Component


class Lineplot(Component):
    def __init__(self, data, x_ticks, id):
        super().__init__()
        self.x_ticks = x_ticks
        self.data = data
        self.id = id

    def render(self) -> html.Div:
        colors = cycle(px.colors.qualitative.Alphabet)
        fig_list = []
        for i, key in enumerate(self.data.keys()):
            color = next(colors)
            key_data = self.data[key]
            mean = np.mean(key_data, axis=0)
            sd = np.std(key_data, axis=0)
            ci_upper = mean + (1.96 * sd / np.sqrt(key_data.shape[0]))
            ci_lower = mean - (1.96 * sd / np.sqrt(key_data.shape[0]))
            fig_list.append(go.Scatter(x=self.x_ticks, y=mean, line=dict(color=color),
                                       legendgroup=key, name=key, mode="lines"))
            rgb_col = px.colors.hex_to_rgb(color)
            fig_list.append(go.Scatter(x=np.concatenate([self.x_ticks, self.x_ticks[::-1]]),
                                       y=np.concatenate([ci_upper, ci_lower[::-1]]), fill="toself",
                                       fillcolor=f"rgba({rgb_col[0]},{rgb_col[1]},{rgb_col[2]},0.2)",
                                       line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
                                       legendgroup=key))
        return html.Div(dcc.Graph(id=self.id, figure=go.Figure(fig_list)))


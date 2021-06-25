import matplotlib
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
from attrbench.lib.stat import wilcoxon_tests
import plotly
import plotly.graph_objects as go
from plotly import express as px
import pandas as pd
from attrbench.suite.dashboard.components import Component
from  matplotlib.colors import LinearSegmentedColormap

class EffectSizePlot(Component):
    def __init__(self, df, baseline_name, id):
        self.df = df
        self.baseline_name = baseline_name
        self.baseline_data = df[baseline_name].to_numpy()
        self.id = id

    def render(self) -> html.Div:
        x, y, error_x = [], [], []
        for col in self.df.columns:
            if col != self.baseline_name:
                col_data = self.df[col].to_numpy() - self.baseline_data
                x.append(col_data.mean())
                y.append(col)
                error_x.append(1.96 * col_data.std() / np.sqrt(len(col_data)))
        return html.Div(
            dcc.Graph(
                id=self.id,
                figure=go.Figure(
                    go.Bar(x=x, y=y,
                           error_x=dict(type="data", array=error_x, visible=True),
                           orientation="h")
                )
            )
        )

def plotly_effect_size_table(dfs):
    pvalues, effect_sizes, invert = {}, {},{}
    nr_methods = dfs[next(iter(dfs))][0].shape[1]
    # colors = np.array(plotly.colors.n_colors('rgb(255, 0, 0)', 'rgb(0, 225, 0)', nr_methods, colortype='rgb'))
    # colors = matplotlib.cm.get_cmap('RdYlGn')
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    colors = cmap(np.linspace(0, 1, nr_methods), bytes=True)
    colors = [plotly.colors.label_rgb(x[:3]) for x in colors]
    for metric_name, (df, inverted) in dfs.items():
        mef_size, mp_val = wilcoxon_tests(df, inverted)
        effect_sizes[metric_name] = mef_size
        pvalues[metric_name] = mp_val
        invert[metric_name]=inverted

    effect_sizes = pd.DataFrame.from_dict(effect_sizes)
    fill_colors = []
    for k in effect_sizes.columns:
        sorted_effect_size =effect_sizes[k].argsort().to_numpy()
        if not invert[k]:
            sorted_colors = [colors[i] for i in sorted_effect_size]
        else:
            sorted_colors = [colors[i] for i in sorted_effect_size[::-1]]
        fill_colors.append(sorted_colors)
    fill_colors = ['rgb(255,255,255)'] + fill_colors

    effect_sizes = effect_sizes.applymap('{:,.3f}'.format)
    table = go.Table(
        header=dict(values=[''] + list(effect_sizes.columns)),
        cells=dict(
            values=[list(effect_sizes.index)] + [effect_sizes[k].to_list() for k in effect_sizes.columns],
            fill_color=fill_colors
        )
    )
    fig = go.Figure(data=table)

    return fig
import dash_html_components as html
from plotly import express as px
import dash_core_components as dcc
from attrbench.suite.dashboard.components import Component
from attrbench.suite.plot import InterMetricCorrelationPlot, InterMethodCorrelationPlot
import base64
from io import BytesIO


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


def sns_intermetric_correlationplot(dfs):
    fig = InterMetricCorrelationPlot(dfs).render(figsize=(10,10))
    plot_img = BytesIO()
    fig.savefig(plot_img, format="png")
    plot_img.seek(0)
    encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
    return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])

def sns_intermethod_correlationplot(dfs):
    fig = InterMethodCorrelationPlot(dfs).render(figsize=(10,10))
    plot_img = BytesIO()
    fig.savefig(plot_img, format="png")
    plot_img.seek(0)
    encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
    return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])
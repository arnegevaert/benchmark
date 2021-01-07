import base64
from io import BytesIO

import dash_html_components as html
import seaborn as sns

from attrbench.suite.dashboard.components import Component


class ClusterMap(Component):
    def __init__(self, df):
        self.df = df

    def render(self) -> html.Div:
        plot = sns.clustermap(self.df)
        plot_img = BytesIO()
        plot.savefig(plot_img, format="png")
        plot_img.seek(0)
        encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
        return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])


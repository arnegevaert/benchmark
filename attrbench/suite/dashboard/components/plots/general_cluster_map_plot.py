import base64
from io import BytesIO

import dash_html_components as html
import pandas as pd
import seaborn as sns

from attrbench.suite.dashboard.components import Component


class GeneralClusterMapPlot(Component):
    def __init__(self, result_obj, aggregate):
        data = {metric_name: {} for metric_name in result_obj.get_metrics()}
        for metric_name in result_obj.get_metrics():
            for method_name in result_obj.get_methods():
                data[metric_name][method_name] = result_obj.data[metric_name][method_name].stack().mean()
        self.df = pd.DataFrame(data)

    def render(self) -> html.Div:
        plot = sns.clustermap(self.df)
        plot_img = BytesIO()
        plot.savefig(plot_img, format="png")
        plot_img.seek(0)
        encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
        return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])


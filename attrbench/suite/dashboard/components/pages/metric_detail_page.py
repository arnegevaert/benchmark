import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from attrbench.suite.dashboard.components.plots import CorrelationMap, BarPlot
from attrbench.suite.dashboard.util import krippendorff_alpha
from attrbench.suite.dashboard.components.plots import EffectSizePlot, PValueTable
from scipy.stats import ttest_rel
import numpy as np
import pandas as pd
from attrbench.suite.dashboard.components.pages import Page

class MetricDetailPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj)
        self.app = app
        self.rendered = {}

        app.callback(Output("plots-div-metric-detail", "children"),
                     [Input("metric-dropdown", "value"),Input('column_select', 'value')])(self._update_method)
        app.callback(Output('column_select', 'options'),
                     Input("metric-dropdown", "value"))(self._update_dropdown)



    def _update_method(self, metric_name, colum_value):
        if metric_name is not None and colum_value is not None:
            contents = []
            contents.append(html.H2(metric_name))
            meta_data = self.result_obj.metadata[metric_name]
            metric_data = self.result_obj.data[metric_name]

            if meta_data["shape"][1] -1 < colum_value:
                # not all metrics have same nr samples. prevent crash when selecting
                # metric when invalid collumn was still selected
                colum_value = meta_data["shape"][1] -1

            ## Krippendorff
            if meta_data["shape"][0]>1:
                data = np.stack(
                    [metric_data[method_name].to_numpy()[:,colum_value]
                     for method_name in self.result_obj.get_methods()],
                    axis=1)
                krip = krippendorff_alpha(np.argsort(data))
                contents.append(html.H4("krippendorff_alpha for column {}: {}".format(colum_value,krip)))
            # Inter-method correlation
                contents.append(html.H2("Inter-method correlations for column {}:".format(colum_value)))
                data = {method_name: metric_data[method_name].loc[:, colum_value]
                        for method_name in self.result_obj.get_methods()}
                plot = CorrelationMap(pd.concat(data, axis=1), id=f"{metric_name}-method-corr-detail")
                contents.append(plot.render())
            # effect-size

            if meta_data["shape"][0] == self.result_obj.num_samples:
                contents.append(html.H2("Effect-size"))
                data = {}
                for method_name in self.result_obj.get_methods():
                    method_data = metric_data[method_name]
                    data[method_name] = method_data.loc[:,colum_value]
                df = pd.concat(data, axis=1)
                contents.append(EffectSizePlot(df, "Random", id=f"{metric_name}-effect-size").render())
                pvalues = []
                for method_name in self.result_obj.get_methods():
                    if method_name != "Random":
                        statistic, pvalue = ttest_rel(df[method_name].to_numpy(), df["Random"].to_numpy())
                        pvalues.append({"method": method_name, "pvalue": pvalue})
                contents.append(PValueTable(pvalues, id=f"table-pvalues-{metric_name}").render())

            return contents
        return f"No metric selected."



        return f"No metric selected."

    def _update_dropdown(self, metric_name):
        meta = self.result_obj.metadata[metric_name]
        return [{'label':i, 'value':i} for i in range(meta["shape"][1])]

    def render(self) -> html.Div:
        return html.Div([
            dbc.FormGroup([
                dcc.Dropdown(
                    id="metric-dropdown",
                    options=[
                        {"label": metric, "value": metric} for metric in self.result_obj.get_metrics()
                    ],
                    placeholder="Select metric..."),
                dcc.Dropdown(
                    id="column_select",
                    placeholder="select_column",)
            ]),
            html.Div(id="plots-div-metric-detail")
        ])
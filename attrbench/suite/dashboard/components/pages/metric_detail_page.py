import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from attrbench.suite.dashboard.components.plots import CorrelationMap, Lineplot
from attrbench.suite.dashboard.util import krippendorff_alpha
from attrbench.suite.dashboard.components.plots import EffectSizePlot, PValueTable
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
from attrbench.suite.dashboard.components.pages import Page


class MetricDetailPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj)
        self.app = app
        self.rendered = {}

        # Callback to update the plots when metric/column dropdown changes
        app.callback(Output("plots-div-metric-detail", "children"),
                     [Input("metric-dropdown", "value"), Input('column_dropdown', 'value')])(self._update_plots)
        # Callback to update column dropdown options when metric dropdown changes
        app.callback(Output('column_dropdown', 'options'),
                     Input("metric-dropdown", "value"))(self._update_column_dropdown)

    def _update_plots(self, metric_name, column_value):
        if metric_name is not None:
            contents = [html.H2(metric_name)]
            metadata = self.result_obj.metadata[metric_name]
            metric_data = self.result_obj.data[metric_name]

            # Calculate Krippendorff alpha values for all columns if possible (if #rows > 1)
            krip_alphas = []
            if metadata["shape"][0] > 1:
                for col in range(metadata["shape"][1]):
                    data = pd.concat([metric_data[method_name][col]
                                      for method_name in self.result_obj.get_methods()], axis=1).to_numpy()
                    krip_alphas.append(krippendorff_alpha(np.argsort(data)))

            # If metric data has multiple columns, display a lineplot of the Krippendorff alpha value
            if metadata["shape"][1] > 1:
                contents.append(html.H2("Krippendorff alpha values"))
                contents.append(Lineplot(data={"alpha": np.array(krip_alphas).reshape(1, -1)}, x_ticks=np.arange(len(krip_alphas)),
                                         id="krip-alphas-lineplot").render())

            if column_value is not None:
                if metadata["shape"][1] <= column_value:
                    # not all metrics have same nr columns. Prevent crash when selecting
                    # metric when invalid column was still selected
                    column_value = metadata["shape"][1] - 1

                ## Krippendorff
                if metadata["shape"][0] > 1:
                    contents.append(
                        html.H4(f"krippendorff_alpha for column {column_value}: {krip_alphas[column_value]}")
                    )
                    # Inter-method correlation
                    contents.append(html.H2(f"Inter-method correlations for column {column_value}:"))
                    data = {method_name: metric_data[method_name].loc[:, column_value]
                            for method_name in self.result_obj.get_methods()}
                    plot = CorrelationMap(pd.concat(data, axis=1), id=f"{metric_name}-method-corr-detail")
                    contents.append(plot.render())

                # effect-size
                if metadata["shape"][0] == self.result_obj.num_samples:
                    contents.append(html.H2("Effect-size"))
                    data = {}
                    for method_name in self.result_obj.get_methods():
                        method_data = metric_data[method_name]
                        data[method_name] = method_data.loc[:, column_value]
                    df = pd.concat(data, axis=1)
                    contents.append(EffectSizePlot(df, "Random", id=f"{metric_name}-effect-size").render())
                    try:
                        pvalues = []
                        for method_name in self.result_obj.get_methods():
                            if method_name != "Random":
                                statistic, pvalue = wilcoxon(df[method_name].to_numpy(), df["Random"].to_numpy(),
                                                             alternative="two-sided")
                                pvalues.append({"method": method_name, "pvalue": pvalue})
                        contents.append(PValueTable(pvalues, id=f"table-pvalues-{metric_name}").render())
                    except:
                        contents.append(html.P("Wilcoxon test failed."))
            return contents
        return f"No metric selected."

    def _update_column_dropdown(self, metric_name):
        if metric_name is not None:
            meta = self.result_obj.metadata[metric_name]
            return [{'label': i, 'value': i} for i in range(meta["shape"][1])]
        return []

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
                    id="column_dropdown",
                    placeholder="Select column...", )
            ]),
            html.Div(id="plots-div-metric-detail")
        ])

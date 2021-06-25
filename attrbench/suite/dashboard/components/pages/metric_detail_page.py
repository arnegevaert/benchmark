import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from attrbench.suite.dashboard.components.plots import plotly_krippendroff_alpha_plot, wilcoxon_summary_plot
from attrbench.suite.dashboard.components.plots import CorrelationMap, Lineplot
from attrbench.suite.dashboard.components.plots.wilcoxon_plot import plotly_wilcoxon_summary_plot
from attrbench.suite.dashboard.util import krippendorff_alpha, get_dfs, get_metric_df
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
                     [Input("metric-dropdown", "value")])(self._update_plots)
        # Callback to update column dropdown options when metric dropdown changes
        # app.callback(Output('column_dropdown', 'options'),
        #              Input("metric-dropdown", "value"))(self._update_column_dropdown)

    def _update_plots(self, metric_name):
        if metric_name is not None:
            contents = [html.H2(metric_name)]

            dfs = get_metric_df(metric_name,self.result_obj,mode="median_dist")
            # Calculate Krippendorff alpha values for all columns if possible (if #rows > 1)
            contents.append(html.H2("Krippendorff"))
            contents.append(dcc.Graph(figure=plotly_krippendroff_alpha_plot(dfs)))

            # Wilcoxon summary plots
            contents.append(html.H2("Wilcoxon summary"))

            contents.append(dcc.Graph(figure=plotly_wilcoxon_summary_plot(dfs)))

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
                        {"label": metric, "value": metric} for metric in self.result_obj.metric_results.keys()
                    ],
                    placeholder="Select metric...")
            ]),
            html.Div(id="plots-div-metric-detail")
        ])

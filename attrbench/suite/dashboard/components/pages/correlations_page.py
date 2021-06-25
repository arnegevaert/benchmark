import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from attrbench.suite.dashboard import util
from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import CorrelationMap, BarPlot
from attrbench.suite.dashboard.components.plots.correlation_map import sns_intermetric_correlationplot, \
    sns_intermethod_correlationplot
from attrbench.suite.dashboard.util import krippendorff_alpha


class CorrelationsPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj)
        self.rendered_contents = html.Div(id="rendered-content-corr")
        self.app = app
        maskers = result_obj.metric_results[next(iter(result_obj.metric_results))].maskers
        activation_functions = result_obj.metric_results[next(iter(result_obj.metric_results))].activation_fns
        self.add_form = dbc.Form([
            dbc.FormGroup([
                dcc.Dropdown(options=[{"label": i, "value": i}
                                      for i in maskers],
                             value='constant',
                             id="masker-dropdown-corr")
                ]),
            dbc.FormGroup([
                dcc.Dropdown(options=[{"label": activation_fn, "value": activation_fn}
                                      for activation_fn in activation_functions],
                             value='linear',
                             id="activation-dropdown-corr")
                ]),
            dbc.Button("Go", color="primary", id="go-btn-corr")
        ])

        self.app.callback(Output("rendered-content-corr","children"),
                          Input('go-btn-corr','n_clicks'),
                          State('masker-dropdown-corr','value'),
                          State('activation-dropdown-corr','value'),
                          prevent_initial_call=True)(self.update)

    def update(self,nclicks,masker_val,activation_val):
        result=[]
        result.append(html.H2("Inter-metric correlations"))
        dfs = util.get_dfs(self.result_obj, mode='raw', masker=masker_val,
                           activation=activation_val)  # add dropdown menu to select mode, masker, activation
        result.append(sns_intermetric_correlationplot(dfs))

        result.append(html.H2("Inter-method correlations"))
        result.append(sns_intermethod_correlationplot(dfs))
        return html.Div(result)

    def render(self) -> html.Div:
        return html.Div([self.add_form, self.rendered_contents])

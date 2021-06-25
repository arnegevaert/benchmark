import matplotlib
import pandas as pd
import numpy as np

import dash_table
import plotly
import plotly.graph_objects as go
from plotly import express as px
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import attrbench.suite.dashboard.util as util
from attrbench.lib.stat import wilcoxon_tests
from attrbench.suite.dashboard.components.pages import Page
from attrbench.suite.dashboard.components.plots import Lineplot, Boxplot
from attrbench.suite.dashboard.components.plots import plotly_krippendroff_alpha_plot
from attrbench.suite.dashboard.components.plots.effect_size_plot import plotly_effect_size_table


class OverviewPage(Page):
    def __init__(self, result_obj,app):
        super().__init__(result_obj)
        self.app=app
        self.rendered_contents = html.Div(id="rendered-content-overv")
        maskers = result_obj.metric_results[next(iter(result_obj.metric_results))].maskers
        activation_functions = result_obj.metric_results[next(iter(result_obj.metric_results))].activation_fns
        self.add_form = dbc.Form([
            dbc.FormGroup([
                dcc.Dropdown(options=[{"label": i, "value": i}
                                      for i in maskers],
                             value='constant',
                             id="masker-dropdown-overv")
                ]),
            dbc.FormGroup([
                dcc.Dropdown(options=[{"label": activation_fn, "value": activation_fn}
                                      for activation_fn in activation_functions],
                             value='linear',
                             id="activation-dropdown-overv")
                ]),
            dbc.Button("Go", color="primary", id="go-btn-overv")
        ])

        self.app.callback(Output("rendered-content-overv","children"),
                          Input('go-btn-overv','n_clicks'),
                          State('masker-dropdown-overv','value'),
                          State('activation-dropdown-overv','value'),
                          prevent_initial_call=True)(self.update)


    def update(self, nclicks,masker_val,activation_val):
        result=[]
        box_plots=[]
        ka={}
        dfs = util.get_dfs(self.result_obj,mode='raw',masker=masker_val,activation=activation_val) # add dropdown menu to select mode, masker, activation

        for metric_name, (df, inverted)  in dfs.items():
            box_plots.append(html.H2(metric_name))
            box_plots.append(Boxplot(df,metric_name+'_box').render())

        result.append(dcc.Graph(figure=plotly_effect_size_table(dfs)))

        # krippendorff
        result.append(html.H2("Krippendorff"))
        result.append(dcc.Graph(figure=plotly_krippendroff_alpha_plot(dfs)))

        result.extend(box_plots)

        return html.Div(result)

    def render(self):
        return html.Div([self.add_form, self.rendered_contents])


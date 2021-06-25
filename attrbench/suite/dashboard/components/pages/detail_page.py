import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly import express as px

from attrbench.suite.dashboard.components.pages import Page
import attrbench.suite.dashboard.util as util


class DetailPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj)
        self.app = app
        self.rendered = {}
        self.methods = self.result_obj.metric_results[list(self.result_obj.metric_results.keys())[0]].method_names
        # Callback for method selection dropdown
        app.callback(Output("plots-div", "children"),
                     Input("method-dropdown", "value"))(self._update_method)

    def _update_method(self, method_name):
        dfs = util.get_dfs(self.result_obj, mode='raw', masker='constant', activation='linear')
        if method_name is not None:
            if method_name not in self.rendered:
                contents = []
                for metric_name in dfs:
                    contents.append(html.H2(metric_name))
                    metric_data = dfs[metric_name][0][method_name]
                    metric_shape = metric_data.shape[0]
                    plot = px.line(metric_data.transpose()) if metric_shape > 1 else px.violin(metric_data)
                    contents.append(dcc.Graph(id=metric_name, figure=plot))
                self.rendered[method_name] = contents
                return contents
            return self.rendered[method_name]
        return f"No method selected."

    def render(self) -> html.Div:
        return html.Div([
            dbc.FormGroup([
                dcc.Dropdown(
                    id="method-dropdown",
                    options=[
                        {"label": method, "value": method} for method in self.methods
                    ],
                    placeholder="Select method...")
            ]),
            html.Div(id="plots-div")
        ])

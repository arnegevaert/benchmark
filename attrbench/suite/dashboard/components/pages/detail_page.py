import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly import express as px

from attrbench.suite.dashboard.components.pages import Page


class DetailPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj)
        self.app = app
        self.rendered = {}

        # Callback for method selection dropdown
        app.callback(Output("plots-div", "children"),
                     Input("method-dropdown", "value"))(self._update_method)

    def _update_method(self, method_name):
        if method_name is not None:
            if method_name not in self.rendered:
                contents = []
                for metric_name in self.result_obj.get_metrics():
                    contents.append(html.H2(metric_name))
                    metric_data = self.result_obj.data[metric_name][method_name]
                    metric_shape = self.result_obj.metadata[metric_name]["shape"]
                    plot = px.line(metric_data.transpose()) if metric_shape[1] > 1 else px.violin(metric_data)
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
                        {"label": method, "value": method} for method in self.result_obj.get_methods()
                    ],
                    placeholder="Select method...")
            ]),
            html.Div(id="plots-div")
        ])

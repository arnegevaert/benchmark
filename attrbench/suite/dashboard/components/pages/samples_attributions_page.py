import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State

from attrbench.suite.dashboard.components import SampleAttributionsComponent
from attrbench.suite.dashboard.components.pages import Page


class SamplesAttributionsPage(Page):
    def __init__(self, result_obj, app):
        super().__init__(result_obj)
        self.app = app
        self.add_form = dbc.Form([
            dbc.FormGroup([
                dcc.Dropdown(options=[{"label": f"Sample {i}", "value": i}
                                      for i in range(self.result_obj.num_samples)],
                             placeholder="Select sample...",
                             id="sample-dropdown")
            ]),
            dbc.FormGroup([
                dcc.Dropdown(options=[{"label": method_name, "value": method_name}
                                      for method_name in self.result_obj.get_methods()],
                             placeholder="Select methods...", multi=True,
                             id="method-dropdown")
            ]),
            dbc.FormGroup(
                dbc.ButtonGroup([
                    dbc.Button("Add", color="primary", id="add-btn"),
                    dbc.Button("Reset", color="danger", id="reset-btn")
                ])
            )
        ])
        self.alert = dbc.Alert("Please select a sample and at least one method", id="inconsistency-alert",
                               is_open=False, dismissable=True, color="danger")
        self.content = html.Div(id="samples-attrs-content")

        # Callback for adding a row or resetting content
        self.app.callback(Output("samples-attrs-content", "children"),
                          Output("inconsistency-alert", "is_open"),
                          Input("add-btn", "n_clicks"),
                          Input("reset-btn", "n_clicks"),
                          State("sample-dropdown", "value"),
                          State("method-dropdown", "value"),
                          State("samples-attrs-content", "children"), prevent_initial_call=True)(self.update_content)

    def update_content(self, add_btn, reset_btn, sample_index, method_names, cur_children):
        changed_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        if changed_id == "add-btn":
            if sample_index is None or not method_names:
                return cur_children, True
            else:
                result = cur_children if cur_children else []
                id = len(result)
                image = self.result_obj.images[sample_index, ...]
                attrs = np.stack([self.result_obj.attributions[method_name][sample_index, ...]
                                  for method_name in method_names], axis=0)
                result.append(dbc.Row([
                    dbc.Col(html.H4(f"Sample index: {sample_index}"), className="col-md-4")
                ], className="mt-5"))
                result.append(SampleAttributionsComponent(image, attrs, id, method_names).render())
                return result, False
        elif changed_id == "reset-btn":
            return [], False

    def render(self) -> html.Div:
        return html.Div([self.add_form, self.alert, self.content])



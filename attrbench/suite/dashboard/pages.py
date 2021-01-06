from attrbench.suite.dashboard.plots import *
from attrbench.suite.dashboard.components import SampleAttributionsComponent
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import numpy as np
from attrbench.suite.dashboard.util import krippendorff_alpha
from dash.dependencies import Input, Output, State
from plotly import express as px
import dash
import dash_table
from dash_table.Format import Format
from scipy.stats import ttest_rel


class Page(Component):
    def __init__(self, result_obj):
        self.result_obj = result_obj

    def render(self):
        raise NotImplementedError()


class OverviewPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            for metric_name in self.result_obj.get_metrics():
                result.append(html.H2(metric_name))
                if len(self.result_obj.metadata[metric_name]["shape"]) > 1:
                    plot = Lineplot(self.result_obj, metric_name)
                else:
                    plot = Boxplot(self.result_obj, metric_name)
                result.append(dcc.Graph(
                    id=metric_name,
                    figure=plot.render()
                ))
            self.rendered_contents = result
            return result
        return self.rendered_contents


class CorrelationsPage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)
        self.rendered_contents = None

    def render(self):
        if not self.rendered_contents:
            result = []
            # Krippendorff Alpha
            result.append(html.H2("Krippendorff Alpha"))
            names, values = [], []
            for metric_name in self.result_obj.get_metrics():
                metric_data = self.result_obj.data[metric_name]
                metric_metadata = self.result_obj.metadata[metric_name]
                if metric_metadata["shape"][0] == self.result_obj.num_samples:
                    names.append(metric_name)
                    if len(metric_metadata["shape"]) > 1:
                        data = np.stack(
                            [metric_data[method_name].mean(axis=1).to_numpy()
                             for method_name in self.result_obj.get_methods()],
                            axis=1)
                        values.append(krippendorff_alpha(np.argsort(data)))
            result.append(dcc.Graph(
                id="krippendorff-alpha",
                figure=BarPlot(values, names).render()
            ))


            # Inter-metric correlation
            result.append(html.H2("Inter-method correlations"))
            for method_name in self.result_obj.get_methods():
                result.append(html.H3(method_name))
                plot = InterMethodCorrelationPlot(self.result_obj, method_name)
                result.append(dcc.Graph(
                    id=f"{method_name}-metric-corr",
                    figure=plot.render()
                ))

            # Inter-method correlation
            result.append(html.H2("Inter-metric correlations"))
            for metric_name in self.result_obj.get_metrics():
                metric_shape = self.result_obj.metadata[metric_name]["shape"]
                if metric_shape[0] == self.result_obj.num_samples:
                    result.append(html.H3(metric_name))
                    plot = InterMetricCorrelationPlot(self.result_obj, metric_name)
                    result.append(dcc.Graph(
                        id=f"{metric_name}-method-corr",
                        figure=plot.render()
                    ))
            self.rendered_contents = result
            return result
        return self.rendered_contents


class ClusteringPage(Page):
    def render(self):
        plot = GeneralClusterMapPlot(self.result_obj, aggregate=True)
        return plot.render()


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
                attrs = np.concatenate([self.result_obj.attributions[method_name][sample_index, ...]
                                        for method_name in method_names])
                result.append(dbc.Row([
                    dbc.Col(html.H4(f"Sample index: {sample_index}"), className="col-md-4")
                ], className="mt-5"))
                result.append(SampleAttributionsComponent(image, attrs, id, method_names).render())
                return result, False
        elif changed_id == "reset-btn":
            return [], False

    def render(self):
        return [self.add_form, self.alert, self.content]


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
                    plot = px.line(metric_data.transpose()) if len(metric_shape) > 1 else px.violin(metric_data)
                    contents.append(dcc.Graph(id=metric_name, figure=plot))
                self.rendered[method_name] = contents
                return contents
            return self.rendered[method_name]
        return f"No method selected."

    def render(self):
        contents = [
            dbc.FormGroup([
                dcc.Dropdown(
                    id="method-dropdown",
                    options=[
                        {"label": method, "value": method} for method in self.result_obj.get_methods()
                    ],
                    placeholder="Select method...")
            ]),
            html.Div(id="plots-div")
        ]
        return contents


class EffectSizePage(Page):
    def __init__(self, result_obj):
        super().__init__(result_obj)

    def render(self):
        result = []
        for metric_name in self.result_obj.get_metrics():
            metric_shape = self.result_obj.metadata[metric_name]["shape"]
            if metric_shape[0] == self.result_obj.num_samples:
                result.append(html.H2(metric_name))
                data = {}
                for method_name in self.result_obj.get_methods():
                    method_data = self.result_obj.data[metric_name][method_name]
                    # Aggregate if necessary
                    data[method_name] = method_data.mean(axis=1) if len(metric_shape) > 1 else method_data[0]
                df = pd.concat(data, axis=1)
                plot = EffectSizePlot(df, "Random")  # TODO choose baseline name
                result.append(dcc.Graph(
                            id=f"{metric_name}-effect-size",
                            figure=plot.render()
                ))
                pvalues = []
                for method_name in self.result_obj.get_methods():
                    if method_name != "Random":
                        statistic, pvalue = ttest_rel(df[method_name].to_numpy(), df["Random"].to_numpy())
                        pvalues.append({"method": method_name, "pvalue": pvalue})
                result.append(dash_table.DataTable(
                    id=f"table-pvalues-{metric_name}",
                    columns=[{"name": "Method", "id": f"method", "type": "text"},
                             {"name": "p-value", "id": f"pvalue", "type": "numeric", "format": Format(precision=3)}],
                    data=pvalues,
                    style_data_conditional=[
                        {"if": {"filter_query": "{pvalue} < 0.05"}, "backgroundColor": "lightgreen"},
                        {"if": {"filter_query": "{pvalue} >= 0.05"}, "backgroundColor": "lightred"},
                    ],
                    style_table={"width": "30%"}
                ))
        return result

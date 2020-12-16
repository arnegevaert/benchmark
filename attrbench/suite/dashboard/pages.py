from attrbench.suite.dashboard.plots import *
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import numpy as np
from attrbench.suite.dashboard.util import krippendorff_alpha
from dash.dependencies import Input, Output
from plotly import express as px
import math


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
        return html.P("Clustering page")


class SamplesAttributionsPage(Page):
    def render(self):
        content = []
        for i in range(self.result_obj.num_samples):
            row_children = []
            # Add original image
            # Put color channel axis last, remove if only 1 channel (squeeze)
            image = np.squeeze(np.moveaxis(self.result_obj.images[i, ...], 0, 2))
            image_fig = px.imshow(image, color_continuous_scale="gray" if len(image.shape) == 2 else None,
                                  width=300, height=300)
            image_fig.update_xaxes(showticklabels=False)
            image_fig.update_yaxes(showticklabels=False)
            image_fig.update_layout(margin=dict(l=0, r=0, t=5, b=5))
            row_children.append(dbc.Col(dcc.Graph(
                id=f"orig-image-{i}",
                figure=image_fig
            ), className="col-md-3"))
            # Add attribution maps
            attrs = np.concatenate([self.result_obj.attributions[method_name][i, ...]
                                    for method_name in self.result_obj.get_methods()])
            attrs_fig = px.imshow(attrs, color_continuous_scale="gray", facet_col=0,
                                  height=300, width=3*300, labels={"facet_col": "method"})
            for j, method_name in enumerate(self.result_obj.get_methods()):
                attrs_fig.layout.annotations[j]["text"] = method_name
            attrs_fig.update_xaxes(showticklabels=False)
            attrs_fig.update_yaxes(showticklabels=False)
            attrs_fig.update_layout(margin=dict(l=0, r=0, t=15, b=0))
            row_children.append(dbc.Col(dcc.Graph(
                id=f"attrs-{i}",
                figure=attrs_fig
            )))

            content.append(dbc.Row(row_children))
        return content


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

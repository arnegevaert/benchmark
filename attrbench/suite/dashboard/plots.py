import pandas as pd
import numpy as np
from plotly import express as px
import plotly.graph_objects as go
from attrbench.suite.dashboard.components import Component
import seaborn as sns
import base64
from io import BytesIO
import dash_html_components as html


class Lineplot(Component):
    def __init__(self, result_obj, metric_name):
        super().__init__()
        self.x_ticks = result_obj.metadata[metric_name]["col_index"]
        self.result_obj = result_obj
        self.data = result_obj.data[metric_name]

    def render(self):
        colors = px.colors.qualitative.Plotly
        fig_list = []
        for i, method_name in enumerate(self.result_obj.get_methods()):
            method_data = self.data[method_name]
            mean = np.mean(method_data, axis=0)
            sd = np.std(method_data, axis=0)
            ci_upper = mean + (1.96 * sd / np.sqrt(method_data.shape[0]))
            ci_lower = mean - (1.96 * sd / np.sqrt(method_data.shape[0]))
            fig_list.append(go.Scatter(x=self.x_ticks, y=mean, line=dict(color=colors[i]), legendgroup=method_name, name=method_name, mode="lines"))
            rgb_col = px.colors.hex_to_rgb(colors[i])
            fig_list.append(go.Scatter(x=np.concatenate([self.x_ticks, self.x_ticks[::-1]]),
                                       y=np.concatenate([ci_upper, ci_lower[::-1]]), fill="toself",
                                       fillcolor=f"rgba({rgb_col[0]},{rgb_col[1]},{rgb_col[2]},0.2)",
                                       line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
                                       legendgroup=method_name))
        return go.Figure(fig_list)


class Boxplot(Component):
    def __init__(self, result_obj, metric_name):
        super().__init__()
        data = result_obj.data[metric_name]
        self.df = pd.concat(data, axis=1)
        self.df.columns = self.df.columns.get_level_values(0)

    def render(self):
        return px.box(self.df)


class InterMethodCorrelationPlot(Component):
    def __init__(self, result_obj, method_name):
        # Take the average metric value for each sample and each metric
        # Only for metrics that have per-sample results (shape[0] > 1)
        data = {metric_name: result_obj.data[metric_name][method_name].mean(axis=1)
                for metric_name in result_obj.get_metrics()
                if result_obj.data[metric_name][method_name].shape[0] > 1}
        self.df = pd.concat(data, axis=1)

    def render(self):
        corrs = self.df.corr(method="spearman")
        return px.imshow(corrs, zmin=-1, zmax=1)


class InterMetricCorrelationPlot(Component):
    def __init__(self, result_obj, metric_name):
        # Take the average metric value for each sample and method, for given metric
        # No need to check shape, this plot should only be called for applicable metrics
        data = {method_name: result_obj.data[metric_name][method_name].mean(axis=1)
                for method_name in result_obj.get_methods()}
        self.df = pd.concat(data, axis=1)

    def render(self):
        corrs = self.df.corr(method="spearman")
        return px.imshow(corrs, zmin=-1, zmax=1)


class BarPlot(Component):
    def __init__(self, values, names):
        self.values = values
        self.names = names

    def render(self):
        return go.Figure([go.Bar(x=self.names, y=self.values)])


class GeneralClusterMapPlot(Component):
    def __init__(self, result_obj, aggregate):
        data = {metric_name: {} for metric_name in result_obj.get_metrics()}
        for metric_name in result_obj.get_metrics():
            for method_name in result_obj.get_methods():
                data[metric_name][method_name] = result_obj.data[metric_name][method_name].stack().mean()
        self.df = pd.DataFrame(data)

    def render(self):
        plot = sns.clustermap(self.df)
        plot_img = BytesIO()
        plot.savefig(plot_img, format="png")
        plot_img.seek(0)
        encoded = base64.b64encode(plot_img.read()).decode("ascii").replace("\n", "")
        return html.Div([html.Img(src=f"data:image/png;base64,{encoded}")])


class MetricClusterMapPlot(Component):
    def render(self):
        # Only applicable to per-sample metrics
        return None  # TODO cluster for single metric (X=samples, Y=methods)


class MethodClusterMapPlot(Component):
    def render(self):
        # Only applicable to per-sample metrics
        return None  # TODO cluster for single method (X=samples, Y=metrics)


class EffectSizePlot(Component):
    def __init__(self, df, baseline_name):
        self.df = df
        self.baseline_name = baseline_name
        self.baseline_data = df[baseline_name].to_numpy()

    def render(self):
        x, y, error_x = [], [], []
        for col in self.df.columns:
            if col != self.baseline_name:
                col_data = self.df[col].to_numpy() - self.baseline_data
                x.append(col_data.mean())
                y.append(col)
                error_x.append(1.96 * col_data.std() / np.sqrt(len(col_data)))
        return go.Figure(
            go.Bar(x=x, y=y,
                   error_x=dict(type="data", array=error_x, visible=True),
                   orientation="h"))

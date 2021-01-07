from attrbench.suite.dashboard.components import Component
import dash_html_components as html


class MetricClusterMapPlot(Component):
    def render(self) -> html.Div:
        # Only applicable to per-sample metrics
        return html.Div()  # TODO cluster for single metric (X=samples, Y=methods)

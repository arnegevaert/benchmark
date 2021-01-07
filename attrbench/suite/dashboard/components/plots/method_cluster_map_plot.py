from attrbench.suite.dashboard.components import Component
import dash_html_components as html


class MethodClusterMapPlot(Component):
    def render(self) -> html.Div:
        # Only applicable to per-sample metrics
        return html.Div()  # TODO cluster for single method (X=samples, Y=metrics)

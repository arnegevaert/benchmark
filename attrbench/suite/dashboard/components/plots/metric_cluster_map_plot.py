from attrbench.suite.dashboard.components import Component


class MetricClusterMapPlot(Component):
    def render(self):
        # Only applicable to per-sample metrics
        return None  # TODO cluster for single metric (X=samples, Y=methods)

from attrbench.suite.dashboard.components import Component


class MethodClusterMapPlot(Component):
    def render(self):
        # Only applicable to per-sample metrics
        return None  # TODO cluster for single method (X=samples, Y=metrics)

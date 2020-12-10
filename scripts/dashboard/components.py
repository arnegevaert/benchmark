class Plot:
    def render(self):
        raise NotImplementedError


class Lineplot(Plot):
    def render(self):
        return None


class Boxplot(Plot):
    def render(self):
        return None


class CorrelationPlot(Plot):
    def render(self):
        return None


class HierarchicalClusteringPlot(Plot):
    def render(self):
        return None


class EffectSizePlot(Plot):
    def render(self):
        return None

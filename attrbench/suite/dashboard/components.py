import pandas as pd
import numpy as np
from plotly import express as px


class Plot:
    def render(self):
        raise NotImplementedError


class Lineplot(Plot):
    def __init__(self, data, x_ticks, normalization=None):
        method_names = list(data.keys())
        self.df = pd.DataFrame(columns=method_names, index=x_ticks)
        self.normalization = normalization
        for method_name in method_names:
            self.df[method_name] = np.average(data[method_name], axis=0)

    def render(self):
        return px.line(self.df)


class Boxplot(Plot):
    def __init__(self, data):
        self.df = pd.concat(data, axis=1)
        self.df.columns = self.df.columns.get_level_values(0)

    def render(self):
        return px.box(self.df)


class CorrelationPlot(Plot):
    def render(self):
        return None


class HierarchicalClusteringPlot(Plot):
    def render(self):
        return None


class EffectSizePlot(Plot):
    def render(self):
        return None


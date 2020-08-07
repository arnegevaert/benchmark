import matplotlib.pyplot as plt
import numpy as np
import json


class _NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# TODO normalization will have to happen via subclasses
class LinePlotResult:
    def __init__(self, data, x_range):
        self.data = data
        self.x_range = x_range
        self.processed = {}
        for method in data:
            method_data = data[method]
            sd = np.std(method_data, axis=0)
            mean = np.mean(method_data, axis=0)
            self.processed[method] = {
                "mean": mean,
                "lower": mean - (1.96 * sd / np.sqrt(method_data.shape[0])),
                "upper": mean + (1.96 * sd / np.sqrt(method_data.shape[0]))
            }

    def plot(self, ci=False, xlog=False, ylog=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        for method in self.processed:
            ax.plot(self.x_range, self.processed[method]["mean"], label=method)
            if ci:
                ax.fill_between(x=self.x_range, y1=self.processed[method]["lower"],
                                y2=self.processed[method]["upper"], alpha=.2)
        ax.legend(loc=(0., 1.05), ncol=3)
        return fig, ax

    def auc(self):
        return {
            method: {
                "mean": np.mean(self.processed[method]["mean"]),
                "lower": np.mean(self.processed[method]["lower"]),
                "upper": np.mean(self.processed[method]["upper"])
            } for method in self.processed
        }

    def save_json(self, filename):
        with open(filename, "w") as outfile:
            json.dump({
                "data": self.data,
                "x_range": self.x_range
            }, outfile, cls=_NumpyJSONEncoder)

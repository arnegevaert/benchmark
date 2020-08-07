import matplotlib.pyplot as plt
import numpy as np
import json


# TODO normalization will have to happen via subclasses
class LinePlotResult:
    def __init__(self, filename=None, raw_data=None):
        if not (raw_data or filename):
            raise ValueError("Must provide raw data dict or file name to load.")
        if raw_data:
            self.processed = {}
            self.x_range = raw_data["x_range"]
            for method in raw_data["data"]:
                method_data = raw_data["data"][method]
                sd = np.std(method_data, axis=0)
                mean = np.mean(method_data, axis=0)
                self.processed[method] = {
                    "mean": mean,
                    "lower": mean - (1.96 * sd / np.sqrt(method_data.shape[0])),
                    "upper": mean + (1.96 * sd / np.sqrt(method_data.shape[0]))
                }
        elif filename:
            with open(filename) as file:
                contents = json.load(file)
                data = contents["data"]
                self.x_range = contents["x_range"]
                self.processed = {
                    method: {
                        stat: np.array(data[method][stat]) for stat in data[method]
                    } for method in data
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

    def save(self, filename):
        with open(filename, "w") as outfile:
            json.dump({
                "x_range": self.x_range,
                "data": {
                    method: {
                        stat: self.processed[method][stat].tolist() for stat in self.processed[method]
                    } for method in self.processed
                }
            }, outfile)

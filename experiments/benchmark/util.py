import os
from os import path
import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(dir):
    result_data = {}
    for filename in os.listdir(dir):
        if path.isdir(path.join(dir, filename)):
            method = filename
            result_data[method] = {}
            metric_files = os.listdir(os.path.join(dir, method))
            for filename in metric_files:
                metric, ext = filename.split('.')
                full_filename = os.path.join(dir, method, filename)
                if ext == "csv":
                    result_data[method][metric] = np.loadtxt(full_filename, delimiter=',')
                elif ext == "json":
                    with open(full_filename) as fp:
                        file_data = json.load(fp)
                        result_data[method][metric] = (np.array(file_data["counts"]) / file_data["total"])
                else:
                    raise ValueError(f"Unrecognized extension {ext} in {method}/{filename}")
        elif os.path.isfile(path.join(dir, filename)):
            if filename == "meta.json":
                with open(os.path.join(dir, filename)) as fp:
                    meta = json.load(fp)
            else:
                raise ValueError(f"Unrecognized file {filename} in {dir}")
    return result_data, meta


def get_metric(data, metric):
    res = {}
    for method in data:
        if metric in data[method]:
            res[method] = data[method][metric]
    return res


def correlation_heatmap(ax, corrs, names, title):
    ax.set_title(title)
    ax.imshow(corrs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corrs[i, j]:.3f}",
                    ha="center", va="center", color="w")

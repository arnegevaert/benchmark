import os
import json
import numpy as np


def load_results(dir):
    result_data = {}
    methods = os.listdir(dir)
    for method in methods:
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
    return result_data


def get_metric(data, metric):
    res = {}
    for method in data:
        if metric in data[method]:
            res[method] = data[method][metric]
    return res

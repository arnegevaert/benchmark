import os
from os import path
import json
import numpy as np


def load_results(dir):
    result_data = {}
    meta = None
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
    data_present = False
    for method in data:
        if metric in data[method]:
            data_present = True
            res[method] = data[method][metric]
    if data_present:
        return res
    return None

import os
from os import path
import numpy as np


class Result:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.metric_names, self.method_names = [], []
        self.data, self.metadata = self.load_results()

    def load_results(self):
        result_data = {}
        meta = {}
        # Files are stored as method > metric
        for method in os.listdir(self.data_dir):
            # If directory, this is method data. If file, it is metadata.
            if path.isdir(path.join(self.data_dir, method)):
                self.method_names.append(method)
                result_data[method] = {}
                metric_files = os.listdir(os.path.join(self.data_dir, method))
                # For each metric file, check extension, store result accordingly (json is impact score)
                for metric_filename in metric_files:
                    metric, ext = metric_filename.split('.')
                    if metric not in self.metric_names:
                        self.metric_names.append(metric)
                    full_filename = os.path.join(self.data_dir, method, metric_filename)
                    result_data[method][metric] = np.loadtxt(full_filename, delimiter=',')
                    # Get header if present
                    with open(full_filename) as fp:
                        header = fp.readline()
                        if header[0] == '#':
                            meta[metric] = [float(x) for x in header[2:].split(',')]
        # Check if any methods have missing data
        for method in self.method_names:
            for metric in self.metric_names:
                if metric not in result_data[method]:
                    print(f"Missing data: {method}/{metric}")
        return result_data, meta

    def get_method(self, method):
        return self.data[method]

    def get_metric(self, metric):
        return {method: self.data[method][metric] for method in self.data if metric in self.data[method]}

    def aggregate(self, method, metric, columns=None):
        # columns: List of column indices to be included in aggregation.
        # Can be used to look at influence of perturbation size.
        data = self.data[method][metric]
        if metric == "deletion":
            return data[:, 0] - np.mean(data[:, columns] if columns is not None else data, axis=1)
        if metric == "insertion":
            return np.mean(data[:, columns] if columns is not None else data, axis=1) - data[:, 0]
        if metric in ["infidelity", "max-sens", "sens-n"]:
            return np.mean(data[:, columns] if columns is not None else data, axis=1)
        if metric in ["impact", "i-coverage", "del-until-flip", "s-impact"]:
            return data

    def bootstrap(self, method, metric, sample_size, num_samples, columns=None):
        data = self.aggregate(method, metric, columns)
        return np.array(
            [np.mean(data[np.random.choice(len(data), sample_size, replace=False)])
             for _ in range(num_samples)])

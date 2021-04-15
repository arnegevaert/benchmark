import numpy as np
from os import path, makedirs
import h5py
from typing import Dict, Optional
from attrbench.metrics import MetricResult
from attrbench import metrics


class SuiteResult:
    def __init__(self, metric_results: Dict[str, MetricResult], num_samples: int, seed: int = None):
        self.metric_results = metric_results
        self.num_samples = num_samples
        self.seed = seed
        self.images: Optional[np.ndarray] = None
        self.attributions: Optional[Dict[str, np.ndarray]] = None

    def add_images(self, images: np.ndarray):
        if self.images is None:
            self.images = images
        else:
            self.images = np.concatenate([self.images, images], axis=0)

    def add_attributions(self, attrs: Dict[str, np.ndarray]):
        if self.attributions is None:
            self.attributions = attrs
        else:
            for method_name in self.attributions.keys():
                self.attributions[method_name] = np.concatenate([self.attributions[method_name], attrs[method_name]])

    def save_hdf(self, filename):
        # if dir not exists: create dir
        if not path.exists(path.dirname(filename)) and path.dirname(filename) != '':
            makedirs(path.dirname(filename))

        with h5py.File(filename, "w") as fp:
            # Save number of samples
            fp.attrs["num_samples"] = self.num_samples
            # Save seed if specified
            if self.seed is not None:
                fp.attrs["seed"] = self.seed
            # Save images if specified
            if self.images is not None:
                fp.create_dataset("images", data=self.images)
            # Save attributions if specified
            if self.attributions:
                attr_group = fp.create_group("attributions")
                for method_name in self.attributions.keys():
                    attr_group.create_dataset(method_name, data=self.attributions[method_name])

            # Save results
            # Each metric gets a group under the "results" group
            result_group = fp.create_group("results")
            for metric_name in self.metric_results:
                metric_group = result_group.create_group(metric_name)
                metric_group.attrs["type"] = str(self.metric_results[metric_name].__class__.__name__)
                self.metric_results[metric_name].add_to_hdf(metric_group)

    @staticmethod
    def load_hdf(filename):
        metric_results = {}
        if path.isfile(filename):
            with h5py.File(filename, "r") as fp:
                images, attributions, seed = None, None, None
                num_samples = fp.attrs["num_samples"]
                if "seed" in fp.attrs.keys():
                    seed = fp.attrs["seed"]
                if "images" in fp.keys():
                    images = np.array(fp["images"])
                if "attributions" in fp.keys():
                    attributions = {
                        method: np.array(fp["attributions"][method])
                        for method in fp["attributions"].keys()
                    }
                result_group = fp["results"]
                for metric_name in result_group.keys():
                    result_type = result_group[metric_name].attrs["type"]
                    result_obj: MetricResult = getattr(metrics, result_type).load_from_hdf(result_group[metric_name])
                    metric_results[metric_name] = result_obj
                return SuiteResult(metric_results, num_samples, seed, images, attributions)
        else:
            raise ValueError(f"File {filename} does not exist")

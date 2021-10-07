import numpy as np
from os import path, makedirs
import h5py
from typing import Dict, Optional
from attrbench.metrics import AbstractMetricResult
from attrbench import metrics


class SuiteResult:
    def __init__(self, metric_results: Optional[Dict[str, AbstractMetricResult]] = None, num_samples: Optional[int] = None,
                 seed: int = None, images: np.ndarray = None, attributions: Dict[str, np.ndarray] = None,
                 predictions: np.ndarray = None):
        self.metric_results = None
        if metric_results is not None:
            self.set_metric_results(metric_results)
        self.num_samples = num_samples
        self.seed = seed
        self.images: Optional[np.ndarray] = images
        self.attributions: Optional[Dict[str, np.ndarray]] = attributions
        self.predictions: Optional[np.ndarray] = predictions

    def set_metric_results(self, metric_results: Dict[str, AbstractMetricResult]):
        self.metric_results = metric_results
        for key, metric_result in self.metric_results.items():
            metric_result.register_suite_result(self)

    def add_images(self, images: np.ndarray):
        if self.images is None:
            self.images = images
        else:
            self.images = np.concatenate([self.images, images], axis=0)

    def add_predictions(self, predictions: np.ndarray):
        if self.predictions is None:
            self.predictions = predictions
        else:
            self.predictions = np.concatenate([self.predictions, predictions], axis=0)

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
            # Save predictions
            if self.predictions is not None:
                fp.create_dataset("predictions", data=self.predictions)

            # Save results
            # Each metric gets a group under the "results" group
            result_group = fp.create_group("results")
            for metric_name in self.metric_results:
                metric_group = result_group.create_group(metric_name)
                result_obj = self.metric_results[metric_name]
                metric_group.attrs["type"] = str(result_obj.__class__.__name__)
                result_obj.add_to_hdf(metric_group)

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
                if "predictions" in fp.keys():
                    predictions = np.array(fp["predictions"])
                if "attributions" in fp.keys():
                    attributions = {
                        method: np.array(fp["attributions"][method])
                        for method in fp["attributions"].keys()
                    }
                result_group = fp["results"]
                for metric_name in result_group.keys():
                    result_type = result_group[metric_name].attrs["type"]
                    result_obj: AbstractMetricResult = getattr(metrics, result_type).load_from_hdf(result_group[metric_name])
                    metric_results[metric_name] = result_obj
                return SuiteResult(metric_results, num_samples, seed, images=images, attributions=attributions,
                                   predictions=predictions)
        else:
            raise ValueError(f"File {filename} does not exist")

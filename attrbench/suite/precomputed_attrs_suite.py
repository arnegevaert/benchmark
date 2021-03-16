from attrbench.suite import SuiteResult
from attrbench.metrics import Metric
from attrbench.lib import AttributionWriter
from .config import Config
from tqdm import tqdm
import torch
import numpy as np
from os import path
from typing import Dict
import logging


class PrecomputedAttrsSuite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """

    def __init__(self, model, attrs: Dict[str, np.ndarray], samples: np.ndarray, batch_size: int, device="cpu",
                 seed=None, log_dir=None, explain_label=None, multi_label=False, num_workers=0):
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.metrics: Dict[str, Metric] = {}
        self.num_workers = num_workers
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.samples_done = 0
        self.seed = seed
        self.log_dir = log_dir
        self.explain_label = explain_label
        self.multi_label = multi_label
        self.attrs = attrs
        self.samples = samples
        self.batch_size = batch_size
        if self.log_dir is not None:
            logging.info(f"Logging TensorBoard to {self.log_dir}")
        self.writer = AttributionWriter(path.join(self.log_dir, "images_and_attributions")) \
            if self.log_dir is not None else None

    def load_config(self, loc):
        global_args = {
            "model": self.model,
            "method_names": list(self.attrs.keys()),
            "num_workers": self.num_workers
        }
        cfg = Config(loc, global_args, log_dir=self.log_dir)
        self.metrics = cfg.load()

    def run(self, num_samples, verbose=True):
        samples_done = 0
        prog = tqdm(total=num_samples) if verbose else None
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        while samples_done < num_samples:
            samples = self.samples[samples_done:samples_done + self.batch_size, ...].to(self.device)
            attrs = {method: self.attrs[method][samples_done:samples_done + self.batch_size, ...]
                     for method in self.attrs.keys()}
            with torch.no_grad():
                out = self.model(samples)
                labels = torch.argmax(out, dim=1)

            # Metric loop
            for i, metric in enumerate(self.attrs.keys()):
                if verbose:
                    prog.set_postfix_str(f"{metric} ({i + 1}/{len(self.metrics)})")
                self.metrics[metric].run_batch(samples, labels, attrs)

            if verbose:
                prog.update(samples.size(0))
            samples_done += samples.size(0)
        self.samples_done += num_samples

    def save_result(self, loc):
        metric_results = {metric_name: self.metrics[metric_name].get_result() for metric_name in self.metrics}
        attrs = None
        if self.save_attrs:
            attrs = {}
            for method_name in self.methods:
                attrs[method_name] = np.concatenate(self.attrs[method_name])
        images = np.concatenate(self.images) if self.save_images else None
        result = SuiteResult(metric_results, self.samples_done, self.seed, images, attrs)
        result.save_hdf(loc)

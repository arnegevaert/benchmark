from attrbench.suite import SuiteResult
from attrbench.metrics import Metric
from attrbench.lib import AttributionWriter
from .config import Config
from tqdm import tqdm
import torch
import numpy as np
from os import path
from typing import Dict
from functools import partial
import multiprocessing


def _run_metric(metric_name, metrics, samples, labels, attrs):
    metrics[metric_name].run_batch(samples, labels, attrs)
    print(f"{metric_name} done.")


_NO_MULTITHREADING = {
    "ImpactCoverage", "MaxSensitivity"
}


class Suite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """

    def __init__(self, model, methods, dataloader, device="cpu",
                 save_images=False, save_attrs=False, seed=None, patch_folder=None,
                 log_dir=None, num_threads=1):
        self.metrics: Dict[str, Metric] = {}
        self.num_threads = num_threads
        if num_threads > 1:
            torch.multiprocessing.set_sharing_strategy("file_system")
        self.model = model.to(device)
        self.model.eval()
        self.methods = methods
        self.dataloader = dataloader
        self.device = device
        self.patch_folder = patch_folder
        self.save_images = save_images
        self.save_attrs = save_attrs
        self.images = []
        self.samples_done = 0
        self.attrs = {method_name: [] for method_name in self.methods}
        self.seed = seed
        self.log_dir = log_dir
        if self.log_dir is not None:
            print(f"Logging TensorBoard to {self.log_dir}")
        self.writer = AttributionWriter(path.join(self.log_dir, "images_and_attributions")) \
            if self.log_dir is not None else None

    def load_config(self, loc):
        cfg = Config(loc, self.log_dir, model=self.model, patch_folder=self.patch_folder, methods=self.methods,
                     method_names=list(self.methods.keys()))
        self.metrics = cfg.load()

    def run(self, num_samples, verbose=True):
        samples_done = 0
        prog = tqdm(total=num_samples) if verbose else None
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        it = iter(self.dataloader)
        # We will check the output shapes of methods on the first batch
        # to make sure they are compatible with the masking policy
        checked_shapes = False
        batch_nr = 0
        while samples_done < num_samples:
            full_batch_cpu, full_labels_cpu = next(it)
            full_batch = full_batch_cpu.to(self.device)
            full_labels = full_labels_cpu.to(self.device)

            # Only use correctly classified samples
            with torch.no_grad():
                pred = torch.argmax(self.model(full_batch), dim=1)
                samples = full_batch[pred == full_labels]
                labels = full_labels[pred == full_labels]

            if samples.size(0) > 0:
                batch_nr += 1
                if samples_done + samples.size(0) > num_samples:
                    diff = num_samples - samples_done
                    samples = samples[:diff]
                    labels = labels[:diff]
                # Save images and attributions if specified
                if self.save_images:
                    self.images.append(samples.cpu().detach().numpy())

                # We need the attributions, to save them or to check their shapes
                attrs = {method_name: self.methods[method_name](samples, labels).cpu().detach().numpy()
                         for method_name in self.methods.keys()}

                if self.writer is not None:
                    self.writer.add_images("Samples", samples, global_step=batch_nr)
                    for name in attrs.keys():
                        self.writer.add_attribution(name, attrs[name], batch_nr)

                # Save attributions if necessary
                if self.save_attrs:
                    for method_name in self.methods:
                        self.attrs[method_name].append(attrs[method_name])

                # Metric loop
                if self.num_threads != 1:
                    # Check which metrics can be run in parallel
                    parallel_metrics = {m: self.metrics[m] for m in self.metrics
                                        if type(self.metrics[m]).__name__ not in _NO_MULTITHREADING}
                    # Use multiprocessing when num_threads > 1
                    with multiprocessing.pool.ThreadPool(self.num_threads) as pool:
                        run_metric_partial = partial(_run_metric, metrics=parallel_metrics, samples=samples,
                                                     labels=labels, attrs=attrs)
                        pool.map(run_metric_partial, parallel_metrics.keys())
                    # Run the metrics that can't be run in parallel
                    non_parallel_metrics = {m: self.metrics[m] for m in self.metrics
                                            if type(self.metrics[m]).__name__ in _NO_MULTITHREADING}
                    for m in non_parallel_metrics.keys():
                        self.metrics[m].run_batch(samples, labels, attrs)
                        print(f"{m} done, not in parallel")
                else:
                    # If num_threads = 1, no multiprocessing is necessary
                    for i, metric in enumerate(self.metrics.keys()):
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

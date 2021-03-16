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
import logging


class Suite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """

    def __init__(self, model, methods, dataloader, device="cpu",
                 save_images=False, save_attrs=False, seed=None, patch_folder=None,
                 log_dir=None, explain_label=None, multi_label=False, num_workers=1):
        self.metrics: Dict[str, Metric] = {}
        self.num_workers = num_workers
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
        self.explain_label = explain_label
        self.multi_label = multi_label
        if self.log_dir is not None:
            logging.info(f"Logging TensorBoard to {self.log_dir}")
        self.writer = AttributionWriter(path.join(self.log_dir, "images_and_attributions")) \
            if self.log_dir is not None else None

    def load_config(self, loc):
        global_args = {
            "model": self.model,
            "patch_folder": self.patch_folder,
            "methods": self.methods,
            "method_names": list(self.methods.keys()),
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
        it = iter(self.dataloader)
        batch_size = self.dataloader.batch_size
        # We will check the output shapes of methods on the first batch
        # to make sure they are compatible with the masking policy
        checked_shapes = False
        batch_nr = 0
        while samples_done < num_samples:
            labels, samples = self.get_batch(it, batch_size)
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
                logging.info("Computing attributions...")
                attrs = {method_name: self.methods[method_name](samples, labels).cpu().detach().numpy()
                         for method_name in self.methods.keys()}
                logging.info("Finished.")

                if self.writer is not None:
                    self.writer.add_images("Samples", samples, global_step=batch_nr)
                    for name in attrs.keys():
                        self.writer.add_attribution(name, attrs[name], batch_nr)

                # Save attributions if necessary
                if self.save_attrs:
                    for method_name in self.methods:
                        self.attrs[method_name].append(attrs[method_name])

                # Metric loop
                for i, metric in enumerate(self.metrics.keys()):
                    if verbose:
                        prog.set_postfix_str(f"{metric} ({i + 1}/{len(self.metrics)})")
                    self.metrics[metric].run_batch(samples, labels, attrs)

                if verbose:
                    prog.update(samples.size(0))
                samples_done += samples.size(0)
        self.samples_done += num_samples

    def get_batch(self, it, batch_size):
        out_samples, out_labels = [], []
        out_size = 0
        # collect a full batch
        while out_size < batch_size:
            full_batch, full_labels = next(it)
            full_batch = full_batch.to(self.device)
            full_labels = full_labels.to(self.device)
            # Only use correctly classified samples
            with torch.no_grad():
                out = self.model(full_batch)
                pred = torch.argmax(out, dim=1)
                if self.multi_label:
                    pred_labels = full_labels[torch.arange(len(pred)), pred]
                    samples = full_batch[pred_labels == 1]
                    labels = pred[pred_labels == 1]
                else:
                    samples = full_batch[pred == full_labels]
                    labels = full_labels[pred == full_labels]
                    if self.explain_label is not None:
                        samples = samples[labels == self.explain_label]
                        labels = labels[labels == self.explain_label]

            out_labels.append(labels.detach().cpu())
            out_samples.append(samples.detach().cpu())
            out_size += samples.size(0)
        labels = torch.cat(out_labels, dim=0)
        samples = torch.cat(out_samples, dim=0)
        return labels[:batch_size].to(self.device), samples[:batch_size].to(self.device)

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

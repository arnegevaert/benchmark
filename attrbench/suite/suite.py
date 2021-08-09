from attrbench.suite import SuiteResult
from attrbench.metrics import Metric
from attrbench.lib import AttributionWriter
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from os import path
from typing import Dict, Callable
import logging


class Suite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """

    def __init__(self, model: torch.nn.Module, methods: Dict[str, Callable], metrics: Dict[str, Metric],
                 device="cpu", num_baseline_samples=25, log_dir: str = None, explain_label: int = None,
                 multi_label=False):
        torch.multiprocessing.set_sharing_strategy("file_system")
        # Save arguments as properties
        self.model = model.to(device)
        self.methods = methods
        self.metrics = metrics
        self.num_baseline_samples = num_baseline_samples
        self.device = device
        self.log_dir = log_dir
        self.explain_label = explain_label
        self.multi_label = multi_label
        self.rng = np.random.default_rng()
        self.attrs_shape = None

        # Construct other properties
        self.model.eval()
        self.writer = None
        if self.log_dir is not None:
            logging.info(f"Logging TensorBoard to {self.log_dir}")
            self.writer = AttributionWriter(path.join(self.log_dir, "images_and_attributions"))

    def _compute_attrs(self, samples, labels):
        logging.info("Computing attributions...")
        attrs = {}
        for method_name in self.methods.keys():
            logging.info(method_name)
            attrs[method_name] = self.methods[method_name](samples, labels).cpu().detach().numpy()
            if self.attrs_shape is None:
                self.attrs_shape = attrs[method_name].shape
        logging.info("Finished.")
        return attrs

    def run(self, dataloader: DataLoader, num_samples: int = None, seed: int = None,
            save_images=False, save_attrs=False, out_filename=None) -> SuiteResult:
        # Initialization
        samples_done = 0
        prog = tqdm(total=num_samples) if num_samples else tqdm()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Initialize result object
        suite_result = SuiteResult(seed=seed)

        it = iter(dataloader)
        batch_size = dataloader.batch_size
        batch_nr = 0
        done = False
        while not done:
            # Check if we need to truncate our final batch
            if num_samples is not None and samples_done + batch_size > num_samples:
                batch_size = num_samples - samples_done

            # Get batch
            try:
                labels, samples, predictions = self.get_batch(it, batch_size)
            except StopIteration:
                break
            batch_nr += 1

            # Calculate all attributions
            attrs_dict = self._compute_attrs(samples, labels)
            baseline_attrs = self.rng.random((self.num_baseline_samples, *self.attrs_shape)) * 2 - 1

            # Save samples to writer
            if self.writer is not None:
                self.writer.add_images("Samples", samples, global_step=batch_nr)
                for name in attrs_dict.keys():
                    self.writer.add_attribution(name, attrs_dict[name], batch_nr)

            # Save images and attributions
            if save_images:
                suite_result.add_images(samples.cpu().detach().numpy())
            if save_attrs:
                suite_result.add_attributions(attrs_dict)

            # Save predictions
            suite_result.add_predictions(predictions)

            # Metric loop
            for i, metric in enumerate(self.metrics.keys()):
                prog.set_postfix_str(f"{metric} ({i + 1}/{len(self.metrics)})")
                self.metrics[metric].run_batch(samples, labels, attrs_dict, baseline_attrs)
            prog.update(samples.size(0))
            samples_done += samples.size(0)

            if out_filename:
                suite_result.set_metric_results({metric: self.metrics[metric].result for metric in self.metrics})
                suite_result.num_samples = samples_done
                suite_result.save_hdf(out_filename)
            # If we have a predetermined amount of samples, check if we have enough
            # Otherwise, we just keep going until there are no samples left
            if num_samples is not None:
                done = samples_done >= num_samples
        return suite_result

    def get_batch(self, it, batch_size):
        out_samples, out_labels, predictions = [], [], []
        out_size = 0
        # collect a full batch of correctly classified samples
        while out_size < batch_size:
            # If the dataset doesn't give any labels, we assume the model correctly classifies them
            # This could be the case if we re-use images from a previous run
            # TODO this only works for multiclass, single label classification
            data = next(it)
            if len(data) == 1:
                samples = data[0]
                out = self.model(samples.to(self.device))
                labels = torch.argmax(out, dim=1)
                binary = False
            else:
                full_batch, full_labels = data
                full_batch = full_batch.to(self.device)
                full_labels = full_labels.to(self.device)

                # Only use correctly classified samples
                with torch.no_grad():
                    out = self.model(full_batch)
                    if out.shape[1] > 1:
                        pred = torch.argmax(out, dim=1)
                        binary = False
                    else:
                        pred = out.squeeze() > 0.  # binary output
                        binary = True
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
            predictions.append(out.detach().cpu())
            out_size += samples.size(0)
        labels = torch.cat(out_labels, dim=0)
        samples = torch.cat(out_samples, dim=0)
        predictions = torch.cat(predictions, dim=0)
        labels = labels[:batch_size].to(self.device) if not binary else torch.zeros_like(labels[:batch_size],
                                                                                         device=self.device)
        samples = samples[:batch_size].to(self.device)
        predictions = predictions[:batch_size].numpy()
        return labels, samples, predictions

from attrbench.suite import metrics, Result
from attrbench.lib import AttributionWriter, masking
from tqdm import tqdm
import torch
import numpy as np
import inspect
from inspect import Parameter
import yaml
from os import path
import warnings


def _parse_masker(d):
    constructor = getattr(masking, d["type"])
    return constructor(**{key: d[key] for key in d if key != "type"})


def _parse_args(args):
    return {key: _parse_masker(args[key]) if key == "masker" else args[key] for key in args}


class Suite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """

    def __init__(self, model, methods, dataloader, device="cpu",
                 save_images=False, save_attrs=False, seed=None, patch_folder=None,
                 log_dir=None, explain_label=None, multi_label=False):
        self.metrics = {}
        self.model = model.to(device)
        self.model.eval()
        self.methods = methods
        self.dataloader = dataloader
        self.device = device
        self.default_args = {"patch_folder": patch_folder, "methods": self.methods}
        self.save_images = save_images
        self.save_attrs = save_attrs
        self.images = []
        self.samples_done = 0
        self.attrs = {method_name: [] for method_name in self.methods}
        self.seed = seed
        self.log_dir = log_dir
        self.explain_label = explain_label
        self.multi_label = multi_label

    def load_config(self, loc):
        with open(loc) as fp:
            data = yaml.full_load(fp)
            # Parse default arguments if present
            default_args = _parse_args(data.get("default", {}))
            self.default_args = {**self.default_args, **default_args}
            # Build metric objects
            for metric_name in data["metrics"]:
                if metric_name in self.metrics.keys():
                    raise ValueError(f"Invalid configuration: duplicate entry {metric_name}")
                metric_dict = data["metrics"][metric_name]
                # Parse args from config file
                args_dict = _parse_args({key: metric_dict[key] for key in metric_dict if key != "type"})
                # Get constructor
                constructor = getattr(metrics, metric_dict["type"])
                # Add model, methods, and (optional) writer args
                args_dict["model"] = self.model
                if self.log_dir:
                    self.writer = AttributionWriter(path.join(self.log_dir, "images_and_attributions"))
                    subdir = path.join(self.log_dir, metric_name)
                    args_dict["writer_dir"] = subdir
                # Compare to required args, add missing ones from default args
                signature = inspect.signature(constructor).parameters
                expected_arg_names = [arg for arg in signature if signature[arg].default == Parameter.empty]
                for e_arg in expected_arg_names:
                    if e_arg not in args_dict:
                        if e_arg in self.default_args:
                            args_dict[e_arg] = self.default_args[e_arg]
                        else:
                            raise ValueError(
                                f"Invalid configuration: required argument {e_arg} not found for metric {metric_name}")
                if metric_dict["type"] == "ImpactCoverage" and self.default_args["patch_folder"] is None:
                    warnings.warn("No patch folder provided, skipping impact coverage.")
                else:
                    # Create metric object using args_dict
                    self.metrics[metric_name] = constructor(**args_dict)

    def run(self, num_samples, verbose=True):
        samples_done = 0
        prog = tqdm(total=num_samples) if verbose else None
        if self.seed:
            torch.manual_seed(self.seed)
        it = iter(self.dataloader)
        batch_size = self.dataloader.batch_size
        # We will check the output shapes of methods on the first batch
        # to make sure they are compatible with the masking policy
        checked_shapes = False
        batch_nr = 0
        while samples_done < num_samples:
            labels, samples = self.get_batch(it, batch_size)

            if samples.size(0) > 0:
                batch_nr +=1
                if samples_done + samples.size(0) > num_samples:
                    diff = num_samples - samples_done
                    samples = samples[:diff]
                    labels = labels[:diff]
                # Save images and attributions if specified
                if self.save_images:
                    self.images.append(samples.cpu().detach().numpy())

                # We need the attributions, to save them or to check their shapes
                attrs = {method_name: self.methods[method_name](samples, labels).cpu().detach()
                         for method_name in self.methods.keys()}
                if self.writer is not None:
                    self.writer.add_image_sample(samples,batch_nr)
                    for name in attrs.keys():
                        self.writer.add_attribution(attrs[name],batch_nr,name)
                if self.save_attrs:
                    # Save attributions if necessary
                    for method_name in self.methods:
                        self.attrs[method_name].append(attrs[method_name].numpy())
                for i, metric in enumerate(self.metrics.keys()):
                    if verbose:
                        prog.set_postfix_str(f"{metric} ({i + 1}/{len(self.metrics)})")
                    if not checked_shapes and hasattr(self.metrics[metric], "masker"):
                        # Check shapes of attributions if necessary
                        for method_name in self.methods:
                            if not self.metrics[metric].masker.check_attribution_shape(samples, attrs[method_name]):
                                raise ValueError(f"Attributions for method {method_name} "
                                                 f"are not compatible with masker")
                        checked_shapes = True
                    self.metrics[metric].run_batch(samples, labels, attrs)
                if verbose:
                    prog.update(samples.size(0))
                samples_done += samples.size(0)
        self.samples_done += num_samples

    def get_batch(self, it, batch_size):
        out_samples, out_labels =[],[]
        out_size =0
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
        labels = torch.cat(out_labels,dim=0)
        samples = torch.cat(out_samples,dim=0)
        return labels[:batch_size].to(self.device), samples[:batch_size].to(self.device)

    def save_result(self, loc):
        data = {
            k: v.get_results()[0]
            for k, v in self.metrics.items()
        }

        meta_data = {}
        for metric_name, metric in self.metrics.items():
            meta_data[metric_name] = metric.metadata
            meta_data[metric_name]["shape"] = metric.get_results()[1]
            meta_data[metric_name]["type"] = type(metric).__name__

        attrs = None
        if self.save_attrs:
            attrs = {}
            for method_name in self.methods:
                attrs[method_name] = np.concatenate(self.attrs[method_name])
        images = np.concatenate(self.images) if self.save_images else None
        res = Result(data, meta_data, num_samples=self.samples_done, seed=self.seed, images=images, attributions=attrs)
        res.save_hdf(loc)

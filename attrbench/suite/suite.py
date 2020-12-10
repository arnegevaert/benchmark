from attrbench.suite import metrics
from attrbench.lib import FeatureMaskingPolicy, PixelMaskingPolicy
from tqdm import tqdm
import torch
import inspect
from inspect import Parameter
import yaml
import h5py


def _parse_masking_policy(d):
    if d["policy"] == "pixel":
        masking_policy = PixelMaskingPolicy(d["value"])
    elif d["policy"] == "feature":
        masking_policy = FeatureMaskingPolicy(d["value"])
    else:
        raise ValueError("policy attribute of masking_policy must be either \"pixel\" or \"feature\"")
    return masking_policy


def _parse_metric_args(metric_args):
    result = {}
    # Fill dictionary with metric-specific arguments from JSON data
    for arg in metric_args:
        if arg == "masking_policy":
            result["masking_policy"] = _parse_masking_policy(metric_args[arg])
        else:
            result[arg] = metric_args[arg]
    return result


def _parse_default_args(default_args):
    return {key: (default_args[key] if key != "masking_policy" else _parse_masking_policy(default_args[key]))
            for key in default_args}


class Suite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """
    def __init__(self, model, methods, dataloader, device="cpu"):
        self.metrics = {}
        self.model = model.to(device)
        self.model.eval()
        self.methods = methods
        self.dataloader = dataloader
        self.device = device
        self.default_args = {}

    def load_config(self, loc):
        with open(loc) as fp:
            data = yaml.full_load(fp)
            # Parse default arguments if present
            self.default_args = _parse_default_args(data.get("default", {}))
            # Build metric objects
            for metric_name in data["metrics"]:
                metric_dict = data["metrics"][metric_name]
                # Parse args from config file
                args_dict = _parse_metric_args(metric_dict.get("args", {}))
                # Get constructor
                constructor = getattr(metrics, metric_dict["type"])
                # Add model and methods args
                args_dict["model"] = self.model
                args_dict["methods"] = self.methods
                # Compare to required args, add missing ones from default args
                signature = inspect.signature(constructor).parameters
                expected_arg_names = [arg for arg in signature if signature[arg].default == Parameter.empty]
                for e_arg in expected_arg_names:
                    if e_arg not in args_dict:
                        if e_arg in self.default_args:
                            args_dict[e_arg] = self.default_args[e_arg]
                        else:
                            raise ValueError(f"Invalid JSON: required argument {e_arg} not found for metric {metric_name}")
                # Create metric object using args_dict
                self.metrics[metric_name] = constructor(**args_dict)

    def run(self, num_samples, verbose=True):
        samples_done = 0
        prog = tqdm(total=num_samples) if verbose else None
        it = iter(self.dataloader)
        while samples_done < num_samples:
            full_batch, full_labels = next(it)
            full_batch = full_batch.to(self.device)
            full_labels = full_labels.to(self.device)

            # Only use correctly classified samples
            pred = torch.argmax(self.model(full_batch), dim=1)
            samples = full_batch[pred == full_labels]
            labels = full_labels[pred == full_labels]
            if samples.size(0) > 0:
                for metric in self.metrics:
                    self.metrics[metric].run_batch(samples, labels)
                if verbose:
                    prog.update(samples.size(0))
                samples_done += samples.size(0)

    def save_result(self, loc):
        with h5py.File(loc, "w") as fp:
            # HDF5 file is laid out as {metric}/{method}
            # Each metric group has the according metadata as attributes
            for metric_name in self.metrics:
                metric = self.metrics[metric_name]
                group = fp.create_group(metric_name)
                # TODO save metadata as attributes for metric
                # x_ticks, normalization method (string)
                """
                for key in meta:
                    group.attrs[key] = meta[key]
                """
                results = metric.get_results()
                for method_name in results:
                    method_results = results[method_name]
                    group.create_dataset(method_name, data=method_results)
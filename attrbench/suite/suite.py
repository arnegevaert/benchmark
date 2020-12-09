import json
from attrbench.suite import metrics
from attrbench.lib import FeatureMaskingPolicy, PixelMaskingPolicy
from tqdm import tqdm
import torch
import inspect
from inspect import Parameter


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
    result = {}
    mp_params = default_args.get("masking_policy", None)
    result["masking_policy"] = _parse_masking_policy(mp_params) if mp_params else None
    result["perturbation_range"] = default_args.get("perturbation_range", None)
    result["mask_range"] = default_args.get("mask_range", None)
    return result


class Suite:
    """
    Represents a "suite" of benchmarking metrics, each with their respective parameters.
    This allows us to very quickly run the benchmark, aggregate and save all the resulting data for
    a given model and dataset.
    """
    def __init__(self, model, methods, dataloader, device="cpu"):
        self.metrics = []
        self.model = model
        self.methods = methods
        self.dataloader = dataloader
        self.device = device
        self.default_args = {}

    def load_json(self, loc):
        with open(loc) as fp:
            data = json.load(fp)
            # Parse default arguments if present
            if "default" in data:
                self.default_args = _parse_default_args(data["default"])
            # Build metric objects
            for metric_name in data["metrics"]:
                # Parse JSON args
                args_dict = _parse_metric_args(data["metrics"][metric_name])
                # Add model and methods args
                args_dict["model"] = self.model
                args_dict["methods"] = self.methods
                # Compare to required args, add missing ones from default args
                constructor = getattr(metrics, metric_name)
                signature = inspect.signature(constructor).parameters
                expected_arg_names = [arg for arg in signature if signature[arg].default == Parameter.empty]
                for e_arg in expected_arg_names:
                    if e_arg not in args_dict:
                        if e_arg in self.default_args:
                            args_dict[e_arg] = self.default_args[e_arg]
                        else:
                            raise ValueError(f"Invalid JSON: required argument {e_arg} not found for metric {metric_name}")
                # Create metric object using args_dict
                self.metrics.append(constructor(**args_dict))

    def run(self, num_samples, verbose=True):
        samples_done = 0
        prog = tqdm(total=num_samples) if verbose else None
        while samples_done < num_samples:
            full_batch, full_labels = next(self.dataloader)
            full_batch = full_batch.to(self.device)
            full_labels = full_labels.to(self.device)

            # Only use correctly classified samples
            pred = torch.argmax(self.model(full_batch), dim=1)
            samples = full_batch[pred == full_labels]
            labels = full_labels[pred == full_labels]
            if samples.size(0) > 0:
                for metric in self.metrics:
                    metric.run_batch(samples, labels)
                if verbose:
                    prog.update(samples.size(0))

    def save_result(self, loc):
        pass  # TODO save to HDF5 file

import json
from attrbench.suite import metrics
from attrbench.lib import FeatureMaskingPolicy, PixelMaskingPolicy
from tqdm import tqdm
import torch


def _parse_masking_policy(d):
    if d["policy"] == "pixel":
        masking_policy = PixelMaskingPolicy(d["value"])
    elif d["policy"] == "feature":
        masking_policy = FeatureMaskingPolicy(d["value"])
    else:
        raise ValueError("policy attribute of masking_policy must be either \"pixel\" or \"feature\"")
    return masking_policy


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
        self.masking_policy = None
        self.perturbation_range = None
        self.mask_range = None

    def add_metric(self, metric: metrics.Metric):
        self.metrics.append(metric)

    def load_json(self, loc):
        with open(loc) as fp:
            data = json.load(fp)
            metric_params = data["metrics"]
            # Configure default parameters
            mp_params = data.get("masking_policy", None)
            self.masking_policy = _parse_masking_policy(mp_params) if mp_params else None
            self.perturbation_range = data.get("perturbation_range", None)
            self.mask_range = data.get("mask_range", None)
            # Configure metric-specific parameters
            for key in metric_params:
                constructor = getattr(metrics, key)
                need_mp = ["Insertion", "Deletion", "ImpactScore", "DeletionUntilFlip", "SensitivityN"]
                # TODO
                if "masking_policy" in metric_params[key]:
                    metric_params[key]["masking_policy"] = _parse_masking_policy(metric_params[key]["masking_policy"])
                if constructor in need_mp:
                    self.add_metric(constructor(self.model, self.methods, self.masking_policy, **metric_params[key]))
                else:
                    self.add_metric(constructor(self.model, self.methods, **metric_params[key]))

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

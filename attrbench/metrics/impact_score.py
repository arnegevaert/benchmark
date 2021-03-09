from attrbench.lib.masking import Masker
from attrbench.metrics import Metric, MetricResult
from typing import Callable, List
import torch
import h5py
import numpy as np
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader


class _ImpactScoreDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        attrs = attrs.reshape(attrs.shape[0], -1)
        self.sorted_indices = np.argsort(attrs)
        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples


def impact_score(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.tensor, num_steps: int,
                 strict: bool, masker: Masker, tau: float = None, writer=None):
    if not (strict or tau):
        raise ValueError("Provide value for tau when calculating non-strict impact score")
    counts = []
    ds = _ImpactScoreDataset(num_steps, samples.cpu().numpy(), attrs, masker)
    dl = DataLoader(ds, shuffle=False, batch_size=1, num_workers=4, pin_memory=True)
    with torch.no_grad():
        orig_out = model(samples)
    batch_size = samples.size(0)
    orig_confidence = softmax(orig_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
    device = samples.device

    for i, masked_samples in enumerate(dl):
        masked_samples = masked_samples[0].to(device).float()
        with torch.no_grad():
            masked_out = model(masked_samples)
        if writer:
            writer.add_images('masked samples', masked_samples, global_step=i)
        confidence = softmax(masked_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
        flipped = torch.argmax(masked_out, dim=1) != labels
        if not strict:
            flipped = flipped | (confidence <= orig_confidence * tau).squeeze()
        counts.append(flipped.sum().item())
    # [len(mask_range)]
    result = torch.tensor(counts)
    return result, batch_size


class ImpactScore(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, strict: bool,
                 masker: Masker, tau: float = None, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.strict = strict
        self.masker = masker
        self.tau = tau
        self.result = ImpactScoreResult(method_names)

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            if method_name not in self.result.method_names:
                raise ValueError(f"Invalid method name: {method_name}")
            flipped, total = impact_score(samples, labels, self.model, attrs_dict[method_name], self.num_steps,
                                          self.strict, self.masker, self.tau, writer=self._get_writer(method_name))
            self.result.append(method_name, flipped, total)


class ImpactScoreResult(MetricResult):
    def __init__(self, method_names: List[str]):
        super().__init__(method_names)
        self.flipped = {m_name: [] for m_name in self.method_names}
        self.totals = {m_name: [] for m_name in self.method_names}

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["type"] = "ImpactScoreResult"
        for method_name in self.method_names:
            flipped = torch.stack(self.flipped[method_name], dim=0).float()
            totals = torch.tensor(self.totals[method_name]).reshape(-1, 1).float()
            method_group = group.create_group(method_name)
            method_group.create_dataset("flipped", flipped.numpy())
            method_group.create_dataset("totals", totals.numpy())

    def append(self, method_name, flipped, total):
        self.flipped[method_name].append(flipped)
        self.totals[method_name].append(total)

    @staticmethod
    def load_from_hdf(self, group: h5py.Group):
        method_names = list(group.keys())
        result = ImpactScoreResult(method_names)
        for m_name in method_names:
            result.flipped[m_name] = group[m_name]["flipped"]
            result.totals[m_name] = group[m_name]["totals"]

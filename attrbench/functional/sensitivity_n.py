from typing import Callable
import numpy as np
from attrbench.lib import mask_segments, segment_samples_attributions
from attrbench.lib.masking import Masker
import torch
import warnings
from torch.utils.data import Dataset, DataLoader


def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                  min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                  masker: Masker, writer=None):
    num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * num_features).astype(np.int)
    ds = SensitivityNDataset(n_range, num_subsets, samples.cpu().numpy(), attrs, masker)
    return _sens_n(samples, labels, attrs, ds, model, n_range, writer)


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: Masker, writer=None):
    # Total number of segments is fixed 100
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * 100).astype(np.int)
    ds = SegSensNDataset(n_range, num_subsets, samples.cpu().numpy(), attrs, masker, writer)
    return _sens_n(samples, labels, attrs, ds, model, n_range, writer)


class SensitivityNDataset(Dataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray, attrs: np.ndarray, masker: Masker):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.attrs = attrs
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.num_features = attrs.reshape(attrs.shape[0], -1).shape[1]

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        indices = np.tile(np.random.choice(self.num_features, size=n, replace=False), (self.samples.shape[0], 1))
        return self.masker.mask(self.samples, indices), indices, n


class SegSensNDataset(Dataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray, attrs: np.ndarray,
                 masker: Masker, writer=None):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.attrs = attrs
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.segmented_images, self.avg_attrs = segment_samples_attributions(samples, attrs)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        valid_indices = [np.where(~np.isinf(self.avg_attrs[i, ...]))[0] for i in range(self.samples.shape[0])]
        indices = np.stack([np.random.choice(valid_indices[i], size=n, replace=False)
                            for i in range(self.samples.shape[0])])
        return mask_segments(self.samples, self.segmented_images, indices, self.masker), indices, n


def _sum_of_attributions(attrs: np.ndarray, indices: np.ndarray):
    attrs = attrs.reshape(attrs.shape[0], -1)
    mask_attrs = np.take_along_axis(attrs, axis=1, indices=indices)
    return mask_attrs.sum(axis=1, keepdims=True)


def _calculate_correlations(sum_of_attrs: np.ndarray, output_diffs: np.ndarray):
    # Calculate correlation between output difference and sum of attribution values
    # Subtract mean
    sum_of_attrs -= sum_of_attrs.mean(axis=1, keepdims=True)
    output_diffs -= output_diffs.mean(axis=1, keepdims=True)
    # Calculate covariances
    cov = (sum_of_attrs * output_diffs).sum(axis=1) / (sum_of_attrs.shape[1] - 1)
    # Divide by product of standard deviations
    # [batch_size]
    denom = sum_of_attrs.std(axis=1) * output_diffs.std(axis=1)
    denom_zero = (denom == 0.)
    if np.any(denom_zero):
        warnings.warn("Zero standard deviation detected.")
    corrcoefs = cov / (sum_of_attrs.std(axis=1) * output_diffs.std(axis=1))
    corrcoefs[denom_zero] = 0.
    return corrcoefs


def _sens_n(samples, labels, attrs, ds, model, n_range, writer=None):
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)
    device = samples.device
    with torch.no_grad():
        orig_output = model(samples)

    output_diffs = {n: [] for n in n_range}
    sum_of_attrs = {n: [] for n in n_range}
    for i, (batch, indices, n) in enumerate(dl):
        batch = batch[0].to(device).float()
        indices = indices[0].numpy()
        n = n.item()
        with torch.no_grad():
            output = model(batch)
        if writer is not None:
            writer.add_images(f"Masked samples N={n}", batch, global_step=i)
        output_diffs[n].append((orig_output - output).gather(dim=1, index=labels.unsqueeze(-1)))
        sum_of_attrs[n].append(_sum_of_attributions(attrs, indices))

    result = []
    for n in n_range:
        n_sum_attrs = np.concatenate(sum_of_attrs[n], axis=1)
        n_out_diffs = torch.cat(output_diffs[n], dim=1).detach().cpu().numpy()
        result.append(_calculate_correlations(n_sum_attrs, n_out_diffs))
    # [batch_size, len(n_range)]
    result = np.stack(result, axis=1)
    return torch.tensor(result)


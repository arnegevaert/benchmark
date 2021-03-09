from typing import Callable, List, Dict
import numpy as np
from attrbench.lib import mask_segments, segment_samples, segment_attributions, AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.lib.util import corrcoef
from attrbench.metrics import Metric, MetricResult
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class _SensitivityNDataset(Dataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray, num_features: int, masker: Masker):
        self.n_range = n_range
        self.num_subsets = num_subsets
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.num_features = num_features

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        indices = np.tile(rng.choice(self.num_features, size=n, replace=False), (self.samples.shape[0], 1))
        return self.masker.mask(self.samples, indices), indices, n


class _SegSensNDataset(_SensitivityNDataset):
    def __init__(self, n_range: np.ndarray, num_subsets: int, samples: np.ndarray,
                 masker: Masker, writer=None):
        super().__init__(n_range, num_subsets, samples, num_features=100, masker=masker)
        self.segmented_images = segment_samples(samples)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        return self.n_range.shape[0] * self.num_subsets

    def __getitem__(self, item):
        n = self.n_range[item // self.num_subsets]
        rng = np.random.default_rng(item)  # Unique seed for each item ensures no duplicate indices
        indices = np.stack([rng.choice(np.unique(self.segmented_images[i, ...]), size=n, replace=False)
                            for i in range(self.samples.shape[0])])
        return mask_segments(self.samples, self.segmented_images, indices, self.masker), indices, n


def _compute_correlations(attrs: np.ndarray, n_range: List[int], output_diffs: Dict[int, np.ndarray],
                          indices: Dict[int, np.ndarray]):
    attrs = attrs.reshape((attrs.shape[0], 1, -1))  # [batch_size, 1, -1]
    result = []
    for n in n_range:
        # Calculate sums of attributions
        n_mask_attrs = np.take_along_axis(attrs, axis=-1, indices=indices[n])  # [batch_size, num_subsets, n]
        n_sum_of_attrs = n_mask_attrs.sum(axis=-1)  # [batch_size, num_subsets]
        n_output_diffs = output_diffs[n]
        # Calculate correlation between output difference and sum of attribution values
        result.append(corrcoef(n_sum_of_attrs, n_output_diffs))

    # [batch_size, len(n_range)]
    result = np.stack(result, axis=1)
    return torch.tensor(result)


def _compute_perturbations(samples: torch.Tensor, labels: torch.Tensor, ds: Dataset,
                           model: Callable, n_range, writer=None):
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)
    device = samples.device
    with torch.no_grad():
        orig_output = model(samples)

    output_diffs = {n: [] for n in n_range}
    removed_indices = {n: [] for n in n_range}
    for i, (batch, indices, n) in enumerate(dl):
        batch = batch[0].to(device).float()
        indices = indices[0].numpy()
        n = n.item()
        with torch.no_grad():
            output = model(batch)
        if writer is not None:
            writer.add_images(f"Masked samples N={n}", batch, global_step=i)
        output_diffs[n].append((orig_output - output).gather(dim=1, index=labels.unsqueeze(-1)))  # [batch_size, 1]
        removed_indices[n].append(indices)  # [batch_size, n]

    for n in n_range:
        output_diffs[n] = torch.cat(output_diffs[n], dim=1).detach().cpu().numpy()  # [batch_size, num_subsets]
        removed_indices[n] = np.stack(removed_indices[n], axis=1)  # [batch_size, num_subsets, n]
    return output_diffs, removed_indices


def sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                  min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                  masker: Masker, writer: AttributionWriter = None):
    num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * num_features).astype(np.int)
    ds = _SensitivityNDataset(n_range, num_subsets, samples.cpu().numpy(), num_features, masker)
    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, writer)
    return _compute_correlations(attrs, n_range, output_diffs, indices)


def seg_sensitivity_n(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
                      min_subset_size: float, max_subset_size: float, num_steps: int, num_subsets: int,
                      masker: Masker, writer: AttributionWriter = None):
    # Total number of segments is fixed 100
    n_range = (np.linspace(min_subset_size, max_subset_size, num_steps) * 100).astype(np.int)
    ds = _SegSensNDataset(n_range, num_subsets, samples.cpu().numpy(), masker, writer)
    attrs = segment_attributions(ds.segmented_images, attrs)
    output_diffs, indices = _compute_perturbations(samples, labels, ds, model, n_range, writer)
    return _compute_correlations(attrs, n_range, output_diffs, indices)


class SensitivityN(Metric):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, masker: Masker, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        self.masker = masker
        self.result = SensitivityNResult(method_names, index=np.linspace(min_subset_size, max_subset_size, num_steps))
        if self.writer_dir is not None:
            self.writers["general"] = AttributionWriter(self.writer_dir)

    def run_batch(self, samples, labels, attrs_dict: Dict[str, np.ndarray]):
        # Get total number of features from attributions dict
        attrs = attrs_dict[next(iter(attrs_dict))]
        num_features = attrs.reshape(attrs.shape[0], -1).shape[1]
        # Calculate n_range
        n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * num_features).astype(np.int)
        # Create pseudo-dataset
        ds = _SensitivityNDataset(n_range, self.num_subsets, samples.cpu().numpy(), num_features, self.masker)
        # Calculate output diffs and removed indices (we will re-use this for each method)
        writer = self.writers["general"] if self.writers is not None else None
        output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, n_range, writer)

        for method_name in attrs_dict:
            attrs = attrs_dict[method_name]
            attrs = attrs.reshape((attrs.shape[0], 1, -1))  # [batch_size, 1, -1]
            self.result.append(method_name, _compute_correlations(attrs, n_range, output_diffs, indices))


class SegSensitivityN(Metric):
    def __init__(self, model: Callable, method_names: List[str], min_subset_size: float, max_subset_size: float,
                 num_steps: int, num_subsets: int, masker: Masker, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.num_steps = num_steps
        self.num_subsets = num_subsets
        # Total number of segments is fixed 100
        self.n_range = (np.linspace(self.min_subset_size, self.max_subset_size, self.num_steps) * 100).astype(np.int)
        self.masker = masker
        self.result = SegSensitivityNResult(method_names, index=np.linspace(min_subset_size, max_subset_size, num_steps))
        if self.writer_dir is not None:
            self.writers["general"] = AttributionWriter(self.writer_dir)

    def run_batch(self, samples, labels, attrs_dict: dict):
        # Create pseudo-dataset
        ds = _SegSensNDataset(self.n_range, self.num_subsets, samples.cpu().numpy(), self.masker)
        # Calculate output diffs and removed indices (we will re-use this for each method)
        writer = self.writers["general"] if self.writers is not None else None
        output_diffs, indices = _compute_perturbations(samples, labels, ds, self.model, self.n_range, writer)

        for method_name in attrs_dict:
            attrs = attrs_dict[method_name]
            attrs = segment_attributions(ds.segmented_images, attrs)
            if self.writers is not None:
                self.writers[method_name].add_images("segmented_attributions", attrs)
            self.result.append(method_name, _compute_correlations(attrs, self.n_range, output_diffs, indices))


class SensitivityNResult(MetricResult):
    def __init__(self, method_names: List[str], index: np.ndarray):
        super().__init__(method_names)
        self.data = {m_name: [] for m_name in self.method_names}
        self.index = index

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["index"] = self.index
        super().add_to_hdf(group)

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        result = cls(method_names, group.attrs["index"])
        result.data = {m_name: np.array(group[m_name]) for m_name in method_names}
        return result


class SegSensitivityNResult(SensitivityNResult):
    pass

from typing import Callable, List, Union, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import Metric, MetricResult


class _InsertionDeletionDataset(Dataset):
    def __init__(self, mode: str, num_steps: int, samples: np.ndarray, attrs: np.ndarray, masker: Masker,
                 reverse_order: bool = False):
        if mode not in ["insertion", "deletion"]:
            raise ValueError("Mode must be insertion or deletion")
        self.mode = mode
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        self.sorted_indices = np.argsort(attrs)
        if reverse_order:
            self.sorted_indices = np.flip(self.sorted_indices, axis=1)

        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:-1].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        indices = self.sorted_indices[:, :-num_to_mask] if self.mode == "insertion" \
            else self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor,
                     model: Callable, ds: _InsertionDeletionDataset, writer=None):
    device = samples.device
    with torch.no_grad():
        orig_preds = model(samples).gather(dim=1, index=labels.unsqueeze(-1))
        fully_masked = torch.tensor(ds.masker.baseline, device=device, dtype=torch.float)
        neutral_preds = model(fully_masked.to(device)).gather(dim=1, index=labels.unsqueeze(-1))
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)

    inter_preds = []
    for i, batch in enumerate(dl):
        batch = batch[0].to(device).float()
        with torch.no_grad():
            predictions = model(batch).gather(dim=1, index=labels.unsqueeze(-1))
        if writer is not None:
            writer.add_images('masked samples', batch, global_step=i)
        inter_preds.append(predictions)
    return orig_preds, neutral_preds, inter_preds


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
              num_steps: int, masker: Masker, reverse_order: bool = False, writer: AttributionWriter = None):
    ds = _InsertionDeletionDataset("insertion", num_steps, samples.cpu().numpy(), attrs, masker, reverse_order)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, writer)
    result = [neutral_preds] + inter_preds + [orig_preds]
    result = torch.cat(result, dim=1)  # [batch_size, len(mask_range)]
    return (result / orig_preds).cpu()


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             num_steps: int, masker: Masker, reverse_order: bool = False, writer: AttributionWriter = None):
    ds = _InsertionDeletionDataset("deletion", num_steps, samples.cpu().numpy(), attrs, masker, reverse_order)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, writer)
    result = [orig_preds] + inter_preds + [neutral_preds]
    result = torch.cat(result, dim=1)  # [batch_size, len(mask_range)]
    return (result / orig_preds).cpu()


class _InsertionDeletion(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], result_class: Callable, method_fn: Callable, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker
        self.modes = (mode,) if type(mode) == str else mode
        self.result = result_class(method_names, self.modes)
        self.method_fn = method_fn

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            method_result = []
            for mode in self.modes:
                reverse_order = mode == "lerf"
                method_result.append(self.method_fn(samples, labels, self.model,
                                                    attrs_dict[method_name], self.num_steps, self.masker,
                                                    writer=self._get_writer(method_name), reverse_order=reverse_order))
            self.result.append(method_name, tuple(method_result))


class Insertion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, masker, mode, InsertionResult, insertion, writer_dir)


class Deletion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, masker, mode, DeletionResult, deletion, writer_dir)


class InsertionDeletionResult(MetricResult):
    def __init__(self, method_names: List[str], modes: Tuple[str]):
        super().__init__(method_names)
        self.modes = modes
        self.data = {m_name: {mode: [] for mode in modes} for m_name in self.method_names}

    def append(self, method_name, batch):
        batch = tuple(batch)
        if len(self.modes) != len(batch):
            raise ValueError(f"Invalid number of results: expected {len(self.modes)}, got {len(batch)}.")
        for i, mode in enumerate(self.modes):
            self.data[method_name][mode].append(batch[i])

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            method_group = group.create_group(method_name)
            for mode in self.modes:
                if type(self.data[method_name][mode]) == list:
                    method_group.create_dataset(mode, data=torch.cat(self.data[method_name][mode]).numpy())
                else:
                    method_group.create_dataset(mode, data=self.data[method_name][mode])

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        modes = tuple(group[method_names[0]].keys())
        result = cls(method_names, modes)
        result.data = {m_name: np.array(group[m_name]) for m_name in method_names}
        return result


class InsertionResult(InsertionDeletionResult):
    def __init__(self, method_names: List[str], modes: Tuple[str]):
        super().__init__(method_names, modes)


class DeletionResult(InsertionDeletionResult):
    def __init__(self, method_names: List[str], modes: Tuple[str]):
        super().__init__(method_names, modes)

from typing import Callable, List, Union, Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import Masker
from attrbench.metrics import Metric, MetricResult
from attrbench.lib.util import ACTIVATION_FNS


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
                     model: Callable, ds: _InsertionDeletionDataset,
                     activation_fns: Tuple[str], writer=None) -> Tuple[Dict, Dict, Dict]:
    device = samples.device
    with torch.no_grad():
        _orig_preds = model(samples)
        orig_preds = {fn: ACTIVATION_FNS[fn](_orig_preds).gather(dim=1, index=labels.unsqueeze(-1))
                      for fn in activation_fns}
        fully_masked = torch.tensor(ds.masker.baseline, device=device, dtype=torch.float)
        _neutral_preds = model(fully_masked.to(device))
        neutral_preds = {fn: ACTIVATION_FNS[fn](_neutral_preds).gather(dim=1, index=labels.unsqueeze(-1))
                         for fn in activation_fns}
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)

    inter_preds = {fn: [] for fn in activation_fns}
    for i, batch in enumerate(dl):
        batch = batch[0].to(device).float()
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images('masked samples', batch, global_step=i)
        for fn in activation_fns:
            inter_preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)))
    return orig_preds, neutral_preds, inter_preds


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
              num_steps: int, masker: Masker, reverse_order: bool = False,
              activation_fn: Union[Tuple[str], str] = "linear", writer: AttributionWriter = None) -> Dict:
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    ds = _InsertionDeletionDataset("insertion", num_steps, samples.cpu().numpy(), attrs, masker, reverse_order)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, activation_fn, writer)
    result = {}
    for fn in activation_fn:
        fn_res = [neutral_preds[fn]] + inter_preds[fn] + [orig_preds[fn]]
        fn_res = torch.cat(fn_res, dim=1)  # [batch_size, len(mask_range)]
        result[fn] = (fn_res / orig_preds[fn]).cpu()
    return result


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             num_steps: int, masker: Masker, reverse_order: bool = False,
             activation_fn: Union[Tuple[str], str] = "linear", writer: AttributionWriter = None) -> Dict:
    ds = _InsertionDeletionDataset("deletion", num_steps, samples.cpu().numpy(), attrs, masker, reverse_order)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, ds, activation_fn, writer)
    result = {}
    for fn in activation_fn:
        fn_res = [orig_preds[fn]] + inter_preds[fn] + [neutral_preds[fn]]
        fn_res = torch.cat(fn_res, dim=1)  # [batch_size, len(mask_range)]
        result[fn] = (fn_res / orig_preds[fn]).cpu()
    return result


class _InsertionDeletion(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 result_class: Callable, method_fn: Callable, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker
        self.modes = (mode,) if type(mode) == str else mode
        self.activation_fn = (activation_fn,) if type(activation_fn) == str else activation_fn
        self.result = result_class(method_names, self.modes, self.activation_fn)
        self.method_fn = method_fn

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            method_result = {}
            for mode in self.modes:
                reverse_order = mode == "lerf"
                method_result[mode] = self.method_fn(samples, labels, self.model,
                                                     attrs_dict[method_name], self.num_steps, self.masker,
                                                     reverse_order, self.activation_fn,
                                                     self._get_writer(method_name))
            self.result.append(method_name, method_result)


class Insertion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, masker, mode, activation_fn,
                         InsertionResult, insertion, writer_dir)


class Deletion(_InsertionDeletion):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str], writer_dir: str = None):
        super().__init__(model, method_names, num_steps, masker, mode, activation_fn,
                         DeletionResult, deletion, writer_dir)


class InsertionDeletionResult(MetricResult):
    def __init__(self, method_names: List[str], modes: Tuple[str], activation_fn: Tuple[str]):
        super().__init__(method_names)
        self.modes = modes
        self.activation_fn = activation_fn
        self.data = {m_name: {mode: {afn: [] for afn in activation_fn} for mode in modes}
                     for m_name in self.method_names}

    def append(self, method_name, batch: Dict):
        for mode in batch.keys():
            for afn in batch[mode].keys():
                self.data[method_name][mode][afn].append(batch[mode][afn])

    def add_to_hdf(self, group: h5py.Group):
        for method_name in self.method_names:
            method_group = group.create_group(method_name)
            for mode in self.modes:
                mode_group = method_group.create_group(mode)
                for afn in self.activation_fn:
                    if type(self.data[method_name][mode][afn]) == list:
                        mode_group.create_dataset(afn, data=torch.cat(self.data[method_name][mode][afn]).numpy())
                    else:
                        mode_group.create_dataset(afn, data=self.data[method_name][mode][afn])

    @classmethod
    def load_from_hdf(cls, group: h5py.Group) -> MetricResult:
        method_names = list(group.keys())
        modes = tuple(group[method_names[0]].keys())
        activation_fn = tuple(group[method_names[0]][modes[0]].keys())
        result = cls(method_names, modes, activation_fn)
        result.data = {
            m_name:
                {mode:
                     {afn: np.array(group[m_name][mode][afn]) for afn in activation_fn}
                 for mode in modes}
            for m_name in method_names
        }
        return result


class InsertionResult(InsertionDeletionResult):
    def __init__(self, method_names: List[str], modes: Tuple[str], activation_fn: Tuple[str]):
        super().__init__(method_names, modes, activation_fn)


class DeletionResult(InsertionDeletionResult):
    def __init__(self, method_names: List[str], modes: Tuple[str], activation_fn: Tuple[str]):
        super().__init__(method_names, modes, activation_fn)

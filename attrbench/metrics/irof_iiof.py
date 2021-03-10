from typing import Callable, List, Tuple, Union, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from attrbench.lib import AttributionWriter
from attrbench.lib import mask_segments, segment_samples_attributions
from attrbench.lib.util import ACTIVATION_FNS
from attrbench.lib.masking import Masker
from attrbench.metrics import Metric, InsertionDeletionResult


class _SegmentedIterativeMaskingDataset(Dataset):
    def __init__(self, mode: str, samples: np.ndarray, attrs: np.ndarray, masker: Masker,
                 reverse_order: bool = False, writer: AttributionWriter = None):
        if mode not in ["insertion", "deletion"]:
            raise ValueError("Mode must be insertion or deletion")
        self.mode = mode
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.segmented_images, avg_attrs = segment_samples_attributions(samples, attrs)
        self.sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]
        if reverse_order:
            self.sorted_indices = np.flip(self.sorted_indices, axis=1)
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        # Exclude fully masked image
        return self.sorted_indices.shape[1] - 1

    def __getitem__(self, item):
        indices = self.sorted_indices[:, :-item] if self.mode == "insertion" else self.sorted_indices[:, -item:]
        return mask_segments(self.samples, self.segmented_images, indices, self.masker)


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor, model: Callable,
                     masking_dataset: _SegmentedIterativeMaskingDataset,
                     activation_fns: Tuple[str], writer=None) -> Tuple[Dict, Dict, Dict]:
    device = samples.device
    with torch.no_grad():
        _orig_preds = model(samples)
        orig_preds = {fn: ACTIVATION_FNS[fn](_orig_preds).gather(dim=1, index=labels.unsqueeze(-1))
                      for fn in activation_fns}
        fully_masked = torch.tensor(masking_dataset.masker.baseline, device=device, dtype=torch.float)
        _neutral_preds = model(fully_masked.to(device))
        neutral_preds = {fn: ACTIVATION_FNS[fn](_neutral_preds).gather(dim=1, index=labels.unsqueeze(-1))
                         for fn in activation_fns}
    masking_dl = DataLoader(masking_dataset, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)

    inter_preds = {fn: [] for fn in activation_fns}
    for i, batch in enumerate(masking_dl):
        batch = batch[0].to(device).float()
        with torch.no_grad():
            predictions = model(batch)
        if writer is not None:
            writer.add_images('masked samples', batch, global_step=i)
        for fn in activation_fns:
            inter_preds[fn].append(ACTIVATION_FNS[fn](predictions).gather(dim=1, index=labels.unsqueeze(-1)))
    return orig_preds, neutral_preds, inter_preds


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, reverse_order: bool = False, activation_fn: Union[Tuple[str], str] = "linear", writer=None):
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    masking_dataset = _SegmentedIterativeMaskingDataset("deletion", samples.cpu().numpy(), attrs, masker,
                                                        reverse_order, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, activation_fn, writer)
    preds = {}
    for fn in activation_fn:
        fn_preds = [orig_preds[fn]] + inter_preds[fn] + [neutral_preds[fn]]
        fn_preds = (torch.cat(fn_preds, dim=1) / orig_preds[fn]).cpu()  # [batch_size, len(mask_range)]
        preds[fn] = fn_preds

    # Calculate AOC for each sample (depends on how many segments each sample had)
    result = {}
    for fn in activation_fn:
        aoc = []
        for i in range(samples.shape[0]):
            num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
            aoc.append(1 - np.trapz(preds[fn][i, :num_segments + 1], x=np.linspace(0, 1, num_segments + 1)))
        result[fn] = torch.tensor(aoc).unsqueeze(-1)
    return result


def iiof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, reverse_order: bool = False, activation_fn: Union[Tuple[str], str] = "linear", writer=None):
    if type(activation_fn) == str:
        activation_fn = (activation_fn,)
    masking_dataset = _SegmentedIterativeMaskingDataset("insertion", samples.cpu().numpy(), attrs, masker,
                                                        reverse_order, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, activation_fn, writer)
    preds = {}
    for fn in activation_fn:
        fn_preds = [neutral_preds[fn]] + inter_preds[fn] + [orig_preds[fn]]
        fn_preds = (torch.cat(fn_preds, dim=1) / orig_preds[fn]).cpu()  # [batch_size, len(mask_range)]
        preds[fn] = fn_preds

    # Calculate AUC for each sample (depends on how many segments each sample had)
    result = {}
    for fn in activation_fn:
        auc = []
        for i in range(samples.shape[0]):
            num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
            auc.append(np.trapz(preds[fn][i, :num_segments + 1], x=np.linspace(0, 1, num_segments + 1)))
        result[fn] = torch.tensor(auc).unsqueeze(-1)
    return result


class _IrofIiof(Metric):
    def __init__(self, model: Callable, method_names: List[str], masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 result_class: Callable, method_fn: Callable, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.masker = masker
        self.modes = (mode,) if type(mode) == str else mode
        for m in self.modes:
            if m not in ("morf", "lerf"):
                raise ValueError(f"Invalid mode: {m}")
        self.activation_fns = (activation_fn,) if type(activation_fn) == str else activation_fn
        self.method_fn = method_fn
        self.result = result_class(method_names, self.modes, self.activation_fns)

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            method_result = {}
            for mode in self.modes:
                reverse_order = mode == "lerf"
                method_result[mode] = self.method_fn(samples, labels, self.model, attrs_dict[method_name],
                                                     self.masker, reverse_order, self.activation_fns,
                                                     writer=self._get_writer(method_name))
            self.result.append(method_name, method_result)


class Irof(_IrofIiof):
    def __init__(self, model: Callable, method_names: List[str], masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, masker, mode, activation_fn, IrofResult, irof, writer_dir)


class Iiof(_IrofIiof):
    def __init__(self, model: Callable, method_names: List[str], masker: Masker,
                 mode: Union[Tuple[str], str], activation_fn: Union[Tuple[str], str],
                 writer_dir: str = None):
        super().__init__(model, method_names, masker, mode, activation_fn, IiofResult, iiof, writer_dir)


class IrofResult(InsertionDeletionResult):
    pass


class IiofResult(InsertionDeletionResult):
    pass

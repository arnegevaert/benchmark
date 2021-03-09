from typing import Callable, List
import h5py
from attrbench.lib.masking import Masker
from attrbench.lib import AttributionWriter
import torch
import numpy as np
from attrbench.lib import mask_segments, segment_samples_attributions
from torch.utils.data import Dataset, DataLoader
from attrbench.metrics import Metric, MetricResult


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
                     masking_dataset: _SegmentedIterativeMaskingDataset, writer=None):
    device = samples.device
    with torch.no_grad():
        orig_preds = model(samples).gather(dim=1, index=labels.unsqueeze(-1))
        fully_masked = torch.tensor(masking_dataset.masker.baseline, device=device, dtype=torch.float)
        neutral_preds = model(fully_masked.to(device)).gather(dim=1, index=labels.unsqueeze(-1))
    masking_dl = DataLoader(masking_dataset, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)

    inter_preds = []
    for i, batch in enumerate(masking_dl):
        batch = batch[0].to(device).float()
        with torch.no_grad():
            predictions = model(batch).gather(dim=1, index=labels.unsqueeze(-1))
        if writer is not None:
            writer.add_images('masked samples', batch, global_step=i)
        inter_preds.append(predictions)
    return orig_preds, neutral_preds, inter_preds


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, reverse_order: bool = False, writer=None):
    masking_dataset = _SegmentedIterativeMaskingDataset("deletion", samples.cpu().numpy(), attrs, masker,
                                                        reverse_order, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, writer)
    preds = [orig_preds] + inter_preds + [neutral_preds]
    preds = (torch.cat(preds, dim=1) / orig_preds).cpu()  # [batch_size, len(mask_range)]

    # Calculate AOC for each sample (depends on how many segments each sample had)
    aoc = []
    for i in range(samples.shape[0]):
        num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
        aoc.append(1 - np.trapz(preds[i, :num_segments + 1], x=np.linspace(0, 1, num_segments + 1)))
    return torch.tensor(aoc).unsqueeze(-1)  # [batch_size, 1]


def iiof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, reverse_order: bool = False, writer=None):
    masking_dataset = _SegmentedIterativeMaskingDataset("insertion", samples.cpu().numpy(), attrs, masker,
                                                        reverse_order, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, writer)
    preds = [neutral_preds] + inter_preds + [orig_preds]
    preds = (torch.cat(preds, dim=1) / orig_preds).cpu()  # [batch_size, len(mask_range)]

    # Calculate AUC for each sample (depends on how many segments each sample had)
    auc = []
    for i in range(samples.shape[0]):
        num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
        auc.append(np.trapz(preds[i, :num_segments + 1], x=np.linspace(0, 1, num_segments + 1)))
    return torch.tensor(auc).unsqueeze(-1)  # [batch_size, 1]


class Irof(Metric):
    def __init__(self, model: Callable, method_names: List[str], masker: Masker,
                 reverse_order: bool = False, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.masker = masker
        self.reverse_order = reverse_order
        self.result = IrofIiofResult(method_names, "irof", reverse_order)

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            method_result = irof(samples, labels, self.model, attrs_dict[method_name], self.masker, self.reverse_order,
                                 writer=self._get_writer(method_name))
            self.result.append(method_name, method_result)


class Iiof(Metric):
    def __init__(self, model: Callable, method_names: List[str], masker: Masker,
                 reverse_order: bool = False, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.masker = masker
        self.reverse_order = reverse_order
        self.result = IrofIiofResult(method_names, "iiof", reverse_order)

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            method_result = iiof(samples, labels, self.model, attrs_dict[method_name], self.masker, self.reverse_order,
                                 writer=self._get_writer(method_name))
            self.result.append(method_name, method_result)


class IrofIiofResult(MetricResult):
    def __init__(self, method_names: List[str], mode: str, reverse_order: bool):
        super().__init__(method_names)
        if mode not in ("irof", "iiof"):
            raise ValueError(f"mode must be either irof or iiof, got {mode}")
        self.data = {m_name: [] for m_name in self.method_names}
        self.mode = mode
        self.reverse_order = reverse_order
        self.inverted = (self.mode == "irof" and not self.reverse_order) or \
                        (self.mode == "iiof" and self.reverse_order)

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["type"] = "IrofIiofResult"
        group.attrs["mode"] = self.mode
        group.attrs["reverse_order"] = self.reverse_order
        for method_name in self.method_names:
            group.create_dataset(method_name, data=torch.cat(self.data[method_name]).numpy())

    def append(self, method_name, batch):
        self.data[method_name].append(batch)

    @staticmethod
    def load_from_hdf(self, group: h5py.Group):
        method_names = list(group.keys())
        result = IrofIiofResult(method_names, group.attrs["mode"], group.attrs["reverse_order"])
        result.data = {m_name: [group[m_name]] for m_name in method_names}

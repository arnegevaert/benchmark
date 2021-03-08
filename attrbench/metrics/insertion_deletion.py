from typing import Callable, List
from attrbench.lib.masking import Masker
from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


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


class Insertion(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 reverse_order: bool = False, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker
        self.reverse_order = reverse_order

    def _run_single_method(self, samples, labels, attrs: np.ndarray, writer=None):
        return insertion(samples, labels, self.model, attrs, self.num_steps, self.masker,
                         writer=writer, reverse_order=self.reverse_order)


class Deletion(Metric):
    def __init__(self, model: Callable, method_names: List[str], num_steps: int, masker: Masker,
                 reverse_order: bool = False, writer_dir: str = None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker
        self.reverse_order = reverse_order

    def _run_single_method(self, samples, labels, attrs: np.ndarray, writer=None):
        return deletion(samples, labels, self.model, attrs, self.num_steps, self.masker,
                        writer=writer, reverse_order=self.reverse_order)


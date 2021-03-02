from typing import Callable
from attrbench.lib.masking import Masker
import torch
import numpy as np
from attrbench.lib import mask_segments, segment_samples_attributions
from torch.utils.data import Dataset, DataLoader


class SegmentedIterativeMaskingDataset(Dataset):
    def __init__(self, mode, samples: np.ndarray, attrs: np.ndarray, masker, writer=None):
        if mode not in ["insertion", "deletion"]:
            raise ValueError("Mode must be insertion or deletion")
        self.mode = mode
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.segmented_images, avg_attrs = segment_samples_attributions(samples, attrs)
        self.sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def __len__(self):
        # Exclude fully masked image
        return self.sorted_indices.shape[1] - 1

    def __getitem__(self, item):
        indices = self.sorted_indices[:, :-item] if self.mode == "insertion" else self.sorted_indices[:, -item:]
        return mask_segments(self.samples, self.segmented_images, indices, self.masker)


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, writer=None):
    masking_dataset = SegmentedIterativeMaskingDataset("deletion", samples.cpu().numpy(), attrs, masker, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, writer)
    preds = [orig_preds] + inter_preds + [neutral_preds]
    preds = (torch.cat(preds, dim=1) / orig_preds).cpu()  # [batch_size, len(mask_range)]

    # Calculate AOC for each sample (depends on how many segments each sample had)
    aoc = []
    for i in range(samples.shape[0]):
        num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
        aoc.append(1 - np.trapz(preds[i, :num_segments+1], x=np.linspace(0, 1, num_segments+1)))
    return torch.tensor(aoc).unsqueeze(-1)  # [batch_size, 1]


def iiof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
         masker: Masker, writer=None):
    masking_dataset = SegmentedIterativeMaskingDataset("insertion", samples.cpu().numpy(), attrs, masker, writer)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, writer)
    preds = [neutral_preds] + inter_preds + [orig_preds]
    preds = (torch.cat(preds, dim=1) / orig_preds).cpu()  # [batch_size, len(mask_range)]

    # Calculate AUC for each sample (depends on how many segments each sample had)
    auc = []
    for i in range(samples.shape[0]):
        num_segments = len(np.unique(masking_dataset.segmented_images[i, ...]))
        auc.append(np.trapz(preds[i, :num_segments+1], x=np.linspace(0, 1, num_segments+1)))
    return torch.tensor(auc).unsqueeze(-1)  # [batch_size, 1]


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor, model: Callable,
                     masking_dataset: SegmentedIterativeMaskingDataset, writer=None):
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


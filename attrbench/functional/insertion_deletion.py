from typing import Callable
from attrbench.lib.masking import Masker
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class IterativeMaskingDataset(Dataset):
    def __init__(self, mode, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
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

        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:-1].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        indices = self.sorted_indices[:, :-num_to_mask] if self.mode == "deletion" \
            else self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
              num_steps: int, masker: Masker, writer=None):
    masking_dataset = IterativeMaskingDataset("insertion", num_steps, samples.cpu().numpy(), attrs, masker)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, writer)
    result = [neutral_preds] + inter_preds + [orig_preds]
    result = torch.cat(result, dim=1)  # [batch_size, len(mask_range)]
    return (result / orig_preds).cpu()


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: np.ndarray,
             num_steps: int, masker: Masker, writer=None):
    masking_dataset = IterativeMaskingDataset("deletion", num_steps, samples.cpu().numpy(), attrs, masker)
    orig_preds, neutral_preds, inter_preds = _get_predictions(samples, labels, model, masking_dataset, writer)
    result = [orig_preds] + inter_preds + [neutral_preds]
    result = torch.cat(result, dim=1)  # [batch_size, len(mask_range)]
    return (result / orig_preds).cpu()


def _get_predictions(samples: torch.Tensor, labels: torch.Tensor,
                     model: Callable, masking_dataset: IterativeMaskingDataset, writer=None):
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

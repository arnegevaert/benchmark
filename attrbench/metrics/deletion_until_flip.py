from typing import Callable
from attrbench.lib.masking import Masker
from attrbench.metrics import Metric
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class _DeletionUntilFlipDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        self.sorted_indices = np.argsort(attrs)
        total_features = attrs.shape[1]
        self.step_size = int(total_features / num_steps)
        if num_steps > total_features or num_steps < 2:
            raise ValueError(f"Number of steps must be between 2 and {total_features} (got {num_steps})")

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        num_to_mask = self.step_size * (item + 1)
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples, num_to_mask


# We assume none of the samples has the same label as the output of the network when given
# a fully masked image (in which case we might not see a flip)
def deletion_until_flip(samples: torch.Tensor, model: Callable, attrs: np.ndarray,
                        num_steps: float, masker: Masker, writer=None):
    if writer is not None:
        flipped_samples = [None for _ in range(samples.shape[0])]
    ds = _DeletionUntilFlipDataset(num_steps, samples.cpu().numpy(), attrs, masker)
    dl = DataLoader(ds, shuffle=False, num_workers=4, pin_memory=True, batch_size=1)
    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()
    device = samples.device

    orig_predictions = torch.argmax(model(samples), dim=1)
    it = iter(dl)
    batch = next(it)
    while not torch.all(flipped) and batch is not None:
        masked_samples, mask_size = batch
        masked_samples = masked_samples[0].to(device).float()
        mask_size = mask_size.item()

        masked_output = model(masked_samples)
        predictions = torch.argmax(masked_output, dim=1)
        criterion = (predictions != orig_predictions)
        new_flipped = torch.logical_or(flipped, criterion.cpu())
        flipped_this_iteration = (new_flipped != flipped)
        if writer is not None:
            for i in range(samples.shape[0]):
                if flipped_this_iteration[i]:
                    flipped_samples[i] = masked_samples[i].cpu()
        result[flipped_this_iteration] = mask_size
        flipped = new_flipped
        try:
            batch = next(it)
        except StopIteration:
            break
    # Set maximum value for samples that were never flipped
    num_inputs = attrs.reshape(attrs.shape[0], -1).shape[1]
    result[result == -1] = num_inputs
    if writer is not None:
        writer.add_images('Flipped samples', torch.stack(
            [s if s is not None else torch.zeros(samples.shape[1:]) for s in flipped_samples]))
    return result


class DeletionUntilFlip(Metric):
    def __init__(self, model, method_names, num_steps, masker, writer_dir=None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker

    def _run_single_method(self, samples, labels, attrs, writer=None):
        return deletion_until_flip(samples, self.model, attrs, self.num_steps,
                                   self.masker, writer=writer).reshape(-1, 1)

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from attrbench.lib.masking import Masker
from attrbench.metrics import Metric
from ._dataset import _DeletionUntilFlipDataset
from .result import DeletionUntilFlipResult


# We assume none of the samples has the same label as the output of the network when given
# a fully masked image (in which case we might not see a flip)
def deletion_until_flip(samples: torch.Tensor, model: Callable, attrs: np.ndarray,
                        num_steps: float, masker: Masker, writer=None):
    if writer is not None:
        flipped_samples = [None for _ in range(samples.shape[0])]
    ds = _DeletionUntilFlipDataset(num_steps, samples.cpu().numpy(), attrs, masker)
    dl = DataLoader(ds, shuffle=False, num_workers=4, batch_size=1)
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
        writer.add_images("flipped samples",
                          torch.stack([s if s is not None else torch.zeros(samples.shape[1:])
                                       for s in flipped_samples]))
    return result.reshape(-1, 1)


class DeletionUntilFlip(Metric):
    def __init__(self, model, method_names, num_steps, masker, writer_dir=None):
        super().__init__(model, method_names, writer_dir)
        self.num_steps = num_steps
        self.masker = masker
        self.result = DeletionUntilFlipResult(method_names)

    def run_batch(self, samples, labels, attrs_dict: dict):
        for method_name in attrs_dict:
            if method_name not in self.result.method_names:
                raise ValueError(f"Invalid method name: {method_name}")
            self.result.append(method_name,
                               deletion_until_flip(samples, self.model, attrs_dict[method_name], self.num_steps,
                                                   self.masker, writer=self._get_writer(method_name))
                               )

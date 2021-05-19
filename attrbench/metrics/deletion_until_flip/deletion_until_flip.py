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
                        num_steps: float, masker: Masker, writer=None, num_workers=0):
    if writer is not None:
        flipped_samples = [None for _ in range(samples.shape[0])]
    ds = _DeletionUntilFlipDataset(num_steps, samples, attrs, masker)
    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()
    orig_predictions = _predict(model,samples)
    it = iter(ds)
    batch = next(it)
    while not torch.all(flipped) and batch is not None:
        masked_samples, mask_size = batch

        predictions = _predict(model,masked_samples)
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


def _predict(model, inputs):
    with torch.no_grad():
        orig_outptus = model(inputs)
        if orig_outptus.shape[1] > 1:
            orig_predictions = torch.argmax(orig_outptus, dim=1)
            binary = False
        else:
            orig_predictions = orig_outptus.squeeze() > 0.
            binary = True
        return orig_predictions

from typing import Callable
from collections import defaultdict

import numpy as np
import torch

from attrbench.lib.masking import Masker
from attrbench.metrics import MaskerMetric
from ._dataset import _MinimalSubsetDataset, _MinimalSubsetDeletionDataset, _MinimalSubsetInsertionDataset
from .result import MinimalSubsetResult


def ms_loop(samples: torch.Tensor, attrs: np.ndarray, ds: _MinimalSubsetDataset, model: Callable,
            orig_predictions: torch.Tensor, flipped: torch.Tensor, result: torch.Tensor, criterion_fn: Callable,
            writer=None):
    flipped_samples = None if writer is None else [None for _ in range(samples.shape[0])]
    it = iter(ds)
    batch = next(it)
    while not torch.all(flipped) and batch is not None:
        masked_samples, mask_size = batch

        with torch.no_grad():
            masked_output = model(masked_samples)
        predictions = torch.argmax(masked_output, dim=1)
        criterion = criterion_fn(predictions, orig_predictions)
        new_flipped = torch.logical_or(flipped, criterion.cpu())
        flipped_this_iteration = (new_flipped != flipped)
        if flipped_samples is not None:
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


def minimal_subset_deletion(samples: torch.Tensor, model: Callable, attrs: np.ndarray,
                            num_steps: float, masker: Masker, writer=None):
    ds = _MinimalSubsetDeletionDataset(num_steps, samples, attrs, masker)
    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()

    orig_predictions = torch.argmax(model(samples), dim=1)

    result = ms_loop(samples, attrs, ds, model, orig_predictions, flipped, result,
                     criterion_fn=lambda pred, orig: pred != orig, writer=writer)
    return result


def minimal_subset_insertion(samples: torch.Tensor, model: Callable, attrs: np.ndarray,
                             num_steps: float, masker: Masker, writer=None):
    flipped_samples = None if writer is None else [None for _ in range(samples.shape[0])]
    ds = _MinimalSubsetInsertionDataset(num_steps, samples, attrs, masker)
    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()

    fully_masked_predictions = torch.argmax(model(ds.masker.baseline), dim=1)
    orig_predictions = torch.argmax(model(samples), dim=1)

    # If the prediction on a fully masked sample (coincidentally) matches the actual prediction,
    # ignore the sample and assign it a score of 0.
    flipped[orig_predictions == fully_masked_predictions] = True
    result[orig_predictions == fully_masked_predictions] = 0
    if writer is not None:
        for i in range(samples.shape[0]):
            if flipped[i]:
                flipped_samples = ds.masker.baseline[i].cpu()

    result = ms_loop(samples, attrs, ds, model, orig_predictions, flipped, result,
                     criterion_fn=lambda pred, orig: pred == orig, writer=writer)
    return result


class _MinimalSubset(MaskerMetric):
    def __init__(self, model, method_names, num_steps, maskers, metric_fn, writer_dir=None):
        super().__init__(model, method_names, maskers, writer_dir)
        self.num_steps = num_steps
        self._result: MinimalSubsetResult = MinimalSubsetResult(method_names + ["_BASELINE"], list(maskers.keys()))
        self.metric_fn = metric_fn

    def run_batch(self, samples, labels, attrs_dict: dict, baseline_attrs: np.ndarray):
        batch_result = defaultdict(dict)
        for masker_name, masker in self.maskers.items():
            # Compute results on baseline attributions
            masker_bl_result = []
            for i in range(baseline_attrs.shape[0]):
                masker_bl_result.append(self.metric_fn(
                    samples, self.model, baseline_attrs[i, ...], self.num_steps,masker
                ).detach().cpu().numpy())
            batch_result[masker_name]["_BASELINE"] = np.stack(masker_bl_result, axis=1)

            # Compute results on actual attributions
            for method_name in attrs_dict:
                batch_result[masker_name][method_name] = self.metric_fn(
                    samples, self.model, attrs_dict[method_name], self.num_steps,
                    masker, writer=self._get_writer(method_name)).detach().cpu().numpy()
        self.result.append(batch_result)


class MinimalSubsetDeletion(_MinimalSubset):
    def __init__(self, model, method_names, num_steps, maskers, writer_dir=None):
        super().__init__(model, method_names, num_steps, maskers, minimal_subset_deletion, writer_dir)


class MinimalSubsetInsertion(_MinimalSubset):
    def __init__(self, model, method_names, num_steps, maskers, writer_dir=None):
        super().__init__(model, method_names, num_steps, maskers, minimal_subset_insertion, writer_dir)

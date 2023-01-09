from attrbench.metrics import MetricWorker
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn
from torch import multiprocessing as mp

from attrbench.masking import Masker
from attrbench.distributed import DoneMessage, PartialResultMessage
from attrbench.metrics.result import BatchResult
from attrbench.data import AttributionsDataset
from ._dataset import _MinimalSubsetDataset, _MinimalSubsetDeletionDataset, _MinimalSubsetInsertionDataset


def ms_loop(attrs: np.ndarray, ds: _MinimalSubsetDataset, model: Callable,
            orig_predictions: torch.Tensor, flipped: torch.Tensor, result: torch.Tensor, criterion_fn: Callable):
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
        result[flipped_this_iteration] = mask_size
        flipped = new_flipped
        try:
            batch = next(it)
        except StopIteration:
            break

    # Set maximum value for samples that were never flipped
    num_inputs = attrs.reshape(attrs.shape[0], -1).shape[1]
    result[result == -1] = num_inputs
    return result.reshape(-1, 1)


def minimal_subset_deletion(samples: torch.Tensor, model: Callable, attrs: np.ndarray,
                            num_steps: float, masker: Masker):
    ds = _MinimalSubsetDeletionDataset(num_steps, samples, attrs, masker)
    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()

    orig_predictions = torch.argmax(model(samples), dim=1)

    result = ms_loop(attrs, ds, model, orig_predictions, flipped, result,
                     criterion_fn=lambda pred, orig: pred != orig)
    return result


def minimal_subset_insertion(samples: torch.Tensor, model: Callable, attrs: np.ndarray,
                             num_steps: float, masker: Masker):
    ds = _MinimalSubsetInsertionDataset(num_steps, samples, attrs, masker)
    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()

    fully_masked_predictions = torch.argmax(model(ds.masker.baseline), dim=1)
    orig_predictions = torch.argmax(model(samples), dim=1)

    # If the prediction on a fully masked sample (coincidentally) matches the actual prediction,
    # ignore the sample and assign it a score of 0.
    flipped[orig_predictions == fully_masked_predictions] = True
    result[orig_predictions == fully_masked_predictions] = 0

    result = ms_loop(attrs, ds, model, orig_predictions, flipped, result,
                     criterion_fn=lambda pred, orig: pred == orig)
    return result



class MinimalSubsetWorker(MetricWorker):
    def __init__(self, result_queue: mp.Queue, rank: int, world_size: int, all_processes_done: mp.Event,
                 model_factory: Callable[[], nn.Module], dataset: AttributionsDataset, batch_size: int,
                 maskers: Dict[str, Masker], mode: str, num_steps: int):
        super().__init__(result_queue, rank, world_size, all_processes_done, model_factory, dataset, batch_size)
        self.maskers = maskers
        self.mode = mode
        self.num_steps = num_steps
        self.metric_fn = minimal_subset_deletion if mode == "deletion" else minimal_subset_insertion

    def work(self):
        model = self._get_model()

        for batch_indices, batch_x, batch_y, batch_attr, method_names in self.dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_result = {}
            for masker_name, masker in self.maskers.items():
                batch_result[masker_name] = self.metric_fn(batch_x, model, batch_attr.detach().cpu().numpy(),
                                                           self.num_steps, masker)
            self.result_queue.put(
                PartialResultMessage(self.rank, BatchResult(batch_indices, batch_result, method_names))
            )
        self.result_queue.put(DoneMessage(self.rank))


from torch.utils.data import DataLoader
from ._dataset import (
    MinimalSubsetDeletionDataset,
    MinimalSubsetInsertionDataset,
)
from attribench.masking import Masker
from typing import Callable, Dict, Mapping
import torch
from torch import nn
from attribench.data import AttributionsDataset
from attribench.result import MinimalSubsetResult
from attribench.result._batch_result import BatchResult
from tqdm import tqdm


def minimal_subset_batch(
    samples: torch.Tensor,
    model: Callable,
    attrs: torch.Tensor,
    num_steps: float,
    maskers: Mapping[str, Masker],
    mode: str,
) -> Dict[str, torch.Tensor]:
    batch_result: Dict[str, torch.Tensor] = {}
    for masker_name, masker in maskers.items():
        if mode == "deletion":
            ds = MinimalSubsetDeletionDataset(
                num_steps, samples, attrs, masker
            )
            criterion_fn = lambda pred, orig: pred != orig
        elif mode == "insertion":
            ds = MinimalSubsetInsertionDataset(
                num_steps, samples, attrs, masker
            )
            criterion_fn = lambda pred, orig: pred == orig
        else:
            raise ValueError("Mode must be deletion or insertion. Got:", mode)

        # Initialize datastructures
        masker_result = torch.tensor(
            [-1 for _ in range(ds.samples.shape[0])]
        ).int()
        flipped = torch.tensor(
            [False for _ in range(ds.samples.shape[0])]
        ).bool()
        orig_predictions = torch.argmax(model(ds.samples), dim=1)

        # The MinimalSubsetDataset is an iterator that returns batches of
        # masked samples and the number of features that were masked.
        it = iter(ds)
        batch = next(it)
        while not torch.all(flipped) and batch is not None:
            masked_samples, mask_size = batch
            # Get output of model on masked samples
            with torch.no_grad():
                masked_output = model(masked_samples)
            predictions = torch.argmax(masked_output, dim=1)

            # Check which samples were flipped to either a different class
            # (deletion) or the original class (insertion)
            criterion = criterion_fn(predictions, orig_predictions)
            new_flipped = torch.logical_or(flipped, criterion.cpu())
            # Record which samples were flipped this iteration
            flipped_this_iteration = new_flipped != flipped
            masker_result[flipped_this_iteration] = mask_size
            flipped = new_flipped
            # Get next batch
            try:
                batch = next(it)
            except StopIteration:
                break

        # Set maximum value for samples that were never flipped
        num_inputs = attrs.reshape(attrs.shape[0], -1).shape[1]
        masker_result[masker_result == -1] = num_inputs
        batch_result[masker_name] = masker_result.reshape(-1, 1)
    return batch_result


def minimal_subset(
    model: nn.Module,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    maskers: Mapping[str, Masker],
    mode: str = "deletion",
    num_steps: int = 100,
    device: torch.device = torch.device("cpu"),
) -> MinimalSubsetResult:
    """Computes Minimal Subset Deletion or Insertion for a given
    :class:`~attribench.data.AttributionsDataset` and model.

    Minimal Subset Deletion or Insertion is computed by iteratively masking
    (Deletion) or revealing (Insertion) the top features of the input samples
    and computing the prediction of the model on the masked samples.

    Minimal Subset Deletion is the minimal number of features that must be
    masked to change the model's prediction from its original prediction.
    Minimal Subset Insertion is the minimal number of features that must be
    revealed to get the model's original prediction.

    The Minimal Subset metric is computed for each masker in `maskers`.

    Parameters
    ----------
    model : nn.Module
        Model to compute the Minimal Subset metric for.
    attributions_dataset : AttributionsDataset
        Dataset containing the samples and attributions to compute
        the Minimal Subset metric for.
    batch_size : int
        Batch size to use when computing the Minimal Subset metric.
    maskers : Dict[str, Masker]
        Dictionary mapping masker names to `Masker` objects.
    mode : str, optional
        "deletion" or "insertion", by default "deletion"
    num_steps : int, optional
        Number of steps to use when computing the Minimal Subset metric,
        by default 100. More steps will result in a more accurate metric,
        but will take longer to compute.
    device : torch.device, optional
        Device to use when computing the Minimal Subset metric,
        by default torch.device("cpu")

    Returns
    -------
    MinimalSubsetResult
    """
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        attributions_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    result = MinimalSubsetResult(
        attributions_dataset.method_names,
        list(maskers.keys()),
        mode,
        num_samples=attributions_dataset.num_samples,
    )

    for (
        batch_indices,
        batch_x,
        _,
        batch_attr,
        method_names,
    ) in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_result = minimal_subset_batch(
            batch_x, model, batch_attr, num_steps, maskers, mode
        )
        result.add(BatchResult(batch_indices, batch_result, method_names))
    return result

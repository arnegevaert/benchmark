from typing import List, Mapping, Optional, Union
from attribench.masking import Masker
from attribench.data import AttributionsDataset
from torch import nn
import torch
from attribench.functional.metrics.deletion._deletion import _deletion_batch
from attribench.result import InsertionResult
from attribench.result._batch_result import BatchResult
from torch.utils.data import DataLoader


def insertion(
    dataset: AttributionsDataset,
    model: nn.Module,
    batch_size: int,
    maskers: Mapping[str, Masker],
    activation_fns: Union[List[str], str] = "linear",
    mode: str = "morf",
    start: float = 0.0,
    stop: float = 1.0,
    num_steps: int = 100,
    device: Optional[torch.device] = None,
):
    """Computes the Insertion metric for a given :class:`~attribench.data.AttributionsDataset` and model.
    Insertion can be viewed as an opposite version of the Deletion metric.

    Insertion is computed by iteratively revealing the top (Most Relevant First,
    or MoRF) or bottom (Least Relevant First, or LeRF) features of
    the input samples, leaving the other features masked out,
    and computing the confidence of the model on the masked
    samples.

    This results in a curve of confidence vs. number of features masked. The
    area under (or equivalently over) this curve is the Insertion metric.

    `start`, `stop`, and `num_steps` are used to determine the range of features
    to mask. The range is determined by `start` and `stop` as a percentage of
    the total number of features. `num_steps` is the number of steps to take
    between `start` and `stop`.

    The Insertion metric is computed for each masker in `maskers` and for each
    activation function in `activation_fns`.

    Parameters
    ----------
    dataset : AttributionsDataset
        Dataset of attributions to compute Insertion on.
    model : nn.Module
        Model to compute Deletion on.
    batch_size : int
        Batch size to use when computing model predictions on masked samples.
    maskers : Dict[str, Masker]
        Dictionary of maskers to use for masking samples.
    activation_fns : Union[List[str], str], optional
        List of activation functions to use when computing model predictions on
        masked samples. If a single string is passed, it is converted to a
        single-element list.
        Default: "linear"
    mode : str, optional
        Mode to use when masking samples. Must be "morf" or "lerf".
        Default: "morf"
    start : float, optional
        Relative start of the range of features to mask. Must be between 0 and 1.
        Default: 0.0
    stop : float, optional
        Relative end of the range of features to mask. Must be between 0 and 1.
        Default: 1.0
    num_steps : int, optional
        Number of steps to use for the range of features to mask.
        Default: 100
    device : Optional[torch.device], optional
        Device to use, by default `None`.
        If `None`, the CPU is used.

    Returns
    -------
    DeletionResult
        Result of the Deletion metric.
    """
    if device is None:
        device = torch.device("cpu")
    if isinstance(activation_fns, str):
        activation_fns = [activation_fns]

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    result = InsertionResult(
        dataset.method_names,
        tuple(maskers.keys()),
        tuple(activation_fns),
        mode,
        shape=(dataset.num_samples, num_steps),
    )

    for (
        batch_indices,
        batch_x,
        batch_y,
        batch_attr,
        method_names,
    ) in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_result = _deletion_batch(
            batch_x,
            batch_y,
            model,
            batch_attr.numpy(),
            maskers,
            activation_fns,
            "morf" if mode == "lerf" else "lerf",  # swap mode
            1 - start,  # swap start
            1 - stop,  # swap stop
            num_steps,
        )
        result.add(BatchResult(batch_indices, batch_result, method_names))
    return result

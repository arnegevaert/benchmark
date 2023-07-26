import torch
from torch import nn
from typing import List, Union, Mapping, Dict
from attribench.masking.image import ImageMasker
from attribench.functional.metrics.deletion._dataset import IrofDataset
from attribench.functional.metrics.deletion._get_predictions import (
    get_predictions,
)
from attribench.data import AttributionsDataset
from torch.utils.data import DataLoader
from attribench.result._deletion_result import DeletionResult
from attribench.result._batch_result import BatchResult


def _irof_batch(
    samples: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    attrs: torch.Tensor,
    maskers: Mapping[str, ImageMasker],
    activation_fns: List[str],
    mode: str,
    start: float,
    stop: float,
    num_steps: int,
) -> Dict:
    result_dict = {}
    for masker_name, masker in maskers.items():
        masking_dataset = IrofDataset(
            mode, start, stop, num_steps, samples, masker
        )
        masking_dataset.set_attrs(attrs)
        result_dict[masker_name] = get_predictions(
            masking_dataset, labels, model, activation_fns
        )
    return result_dict


def irof(
    model: nn.Module,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    maskers: Mapping[str, ImageMasker],
    activation_fns: Union[List[str], str] = "linear",
    mode: str = "morf",
    start: float = 0.0,
    stop: float = 1.0,
    num_steps: int = 100,
    device: torch.device = torch.device("cpu"),
):
    """Computes the IROF metric for a given :class:`~attribench.data.AttributionsDataset` and model.

    IROF starts segmenting the input image using SLIC. Then, it iteratively
    masks out the top (Most Relevant First, or MoRF) or bottom (Least Relevant
    First, or LeRF) segments and computes the confidence of the model on the
    masked samples. The relevance of a segment is computed as the average
    relevance of the features in the segment.

    This results in a curve of confidence vs. number of segments masked. The
    area under (or equivalently over) this curve is the IROF metric.

    `start`, `stop`, and `num_steps` are used to determine the range of segments
    to mask. The range is determined by `start` and `stop` as a percentage of
    the total number of segments. `num_steps` is the number of steps to take
    between `start` and `stop`.

    The IROF metric is computed for each masker in `maskers` and for each
    activation function in `activation_fns`.

    Parameters
    ----------
    model : nn.Module
        Model to compute IROF on.
    attributions_dataset : AttributionsDataset
        Dataset of attributions to compute IROF on.
    batch_size : int
        Batch size to use when computing model predictions on masked samples.
    maskers : Mapping[str, ImageMasker]
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
        Relative start of the range of segments to mask. Must be between 0 and 1.
        Default: 0.0
    stop : float, optional
        Relative stop of the range of segments to mask. Must be between 0 and 1.
        Default: 1.0
    num_steps : int, optional
        Number of steps to take between `start` and `stop`.
        Default: 100
    """
    if isinstance(activation_fns, str):
        activation_fns = [activation_fns]

    model.to(device)
    model.eval()

    dataloader = DataLoader(
        attributions_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    result = DeletionResult(
        attributions_dataset.method_names,
        list(maskers.keys()),
        activation_fns,
        mode,
        num_samples=attributions_dataset.num_samples,
        num_steps=num_steps,
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
        batch_result = _irof_batch(
            batch_x,
            batch_y,
            model,
            batch_attr,
            maskers,
            activation_fns,
            mode,
            start,
            stop,
            num_steps,
        )
        result.add(BatchResult(batch_indices, batch_result, method_names))

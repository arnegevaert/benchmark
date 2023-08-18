import torch
from tqdm import tqdm
from typing import Callable, List, Mapping, Dict, Union
from attribench.masking import Masker
from attribench.data import AttributionsDataset
from torch import nn
from ._dataset import DeletionDataset
from ._get_predictions import get_predictions
from torch.utils.data import DataLoader
from attribench.result import DeletionResult
from attribench.result._batch_result import BatchResult


def _deletion_batch(
    samples: torch.Tensor,
    labels: torch.Tensor,
    model: Callable,
    attrs: torch.Tensor,
    maskers: Mapping[str, Masker],
    activation_fns: List[str],
    mode: str,
    start: float,
    stop: float,
    num_steps: int,
) -> Dict:
    result_dict = {}
    for masker_name, masker in maskers.items():
        ds = DeletionDataset(
            mode, start, stop, num_steps, samples, attrs, masker
        )
        result_dict[masker_name] = get_predictions(
            ds, labels, model, activation_fns
        )
    return result_dict


def deletion(
    model: nn.Module,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    maskers: Mapping[str, Masker],
    activation_fns: Union[List[str], str] = "linear",
    mode: str = "morf",
    start: float = 0.0,
    stop: float = 1.0,
    num_steps: int = 100,
    device: torch.device = torch.device("cpu"),
) -> DeletionResult:
    """Computes the Deletion metric for a given :class:`~attribench.data.AttributionsDataset` and model.

    Deletion is computed by iteratively masking the top (Most Relevant First,
    or MoRF) or bottom (Least Relevant First, or LeRF) features of
    the input samples and computing the confidence of the model on the masked
    samples.

    This results in a curve of confidence vs. number of features masked. The
    area under (or equivalently over) this curve is the Deletion metric.

    `start`, `stop`, and `num_steps` are used to determine the range of features
    to mask. The range is determined by `start` and `stop` as a percentage of
    the total number of features. `num_steps` is the number of steps to take
    between `start` and `stop`.

    The Deletion metric is computed for each masker in `maskers` and for each
    activation function in `activation_fns`.

    Parameters
    ----------
    model : nn.Module
        Model to compute Deletion on.
    dataset : AttributionsDataset
        Dataset of attributions to compute Deletion on.
    batch_size : int
        Batch size to use when computing model predictions on masked samples.
    maskers : Mapping[str, Masker]
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
    device : torch.device, optional
        Device to use, by default `torch.device("cpu")`

    Returns
    -------
    DeletionResult
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
        attributions_dataset.num_samples,
        num_steps,
    )

    for (
        batch_indices,
        batch_x,
        batch_y,
        batch_attr,
        method_names,
    ) in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_result = _deletion_batch(
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
    return result

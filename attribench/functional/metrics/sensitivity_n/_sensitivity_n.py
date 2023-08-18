import numpy as np
from torch import nn
import torch
import numpy.typing as npt
from typing import Callable, List, Mapping, Dict, Tuple
from attribench.masking import Masker
from attribench.masking.image import ImageMasker
from torch.utils.data import DataLoader
from attribench.data import AttributionsDataset
from attribench._activation_fns import ACTIVATION_FNS
from ._dataset import SensitivityNDataset, SegSensNDataset
from attribench._segmentation import segment_attributions
from attribench._stat import rowwise_pearsonr
from attribench.result import SensitivityNResult
from attribench.result._grouped_batch_result import GroupedBatchResult
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
)
from tqdm import tqdm


def _get_orig_output(
    samples: torch.Tensor, model: Callable, activation_fns: List[str]
):
    activated_orig_output = {}
    with torch.no_grad():
        orig_output = model(samples)
        for activation_fn in activation_fns:
            activated_orig_output[activation_fn] = ACTIVATION_FNS[
                activation_fn
            ](orig_output)
    return activated_orig_output


def _compute_out_diffs(
    model: Callable,
    ds: SensitivityNDataset | SegSensNDataset,
    activation_fns: List[str],
    orig_output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> Tuple[Dict[str, Dict[int, npt.NDArray]], Dict[int, npt.NDArray]]:
    n_range = ds.n_range
    output_diff_shape = (ds.samples.shape[0], ds.num_subsets)
    # Calculate differences in output and removed indices
    # (will be re-used for all methods)
    # activation_fn -> n -> [batch_size, num_subsets]
    output_diffs: Dict[str, Dict[int, npt.NDArray]] = {
        activation_fn: {n: np.zeros(output_diff_shape) for n in n_range}
        for activation_fn in activation_fns
    }
    removed_indices: Dict[int, npt.NDArray] = {
        n: np.zeros((ds.samples.shape[0], ds.num_subsets, n), dtype=int)
        for n in n_range
    }
    # TODO why do we not use a dataloader here?
    for i in range(len(ds)):
        batch, indices, n, subset_idx = ds[i]
        n = n.item()
        with torch.no_grad():
            output = model(batch)
        for activation_fn in activation_fns:
            activated_output = ACTIVATION_FNS[activation_fn](output)
            # [batch_size, 1]
            output_diffs[activation_fn][n][:, subset_idx] = (
                (orig_output[activation_fn] - activated_output)
                .gather(dim=1, index=labels.unsqueeze(-1))
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )
        removed_indices[n][:, subset_idx, :] = indices  # [batch_size, n]
    return output_diffs, removed_indices


def _compute_correlations(
    method_names: List[str],
    batch_attr: Dict[str, torch.Tensor],
    ds: SensitivityNDataset | SegSensNDataset,
    segmented: bool,
    removed_indices: Dict[int, npt.NDArray],
    output_diffs: Dict[str, Dict[int, npt.NDArray]],
    activation_fns: List[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    # activation_fn -> method_name -> [batch_size, len(n_range)]
    result = {
        activation_fn: {
            method_name: torch.zeros((ds.samples.shape[0], len(ds.n_range)))
            for method_name in method_names
        }
        for activation_fn in activation_fns
    }
    # Compute correlations for all methods
    # TODO can we use joblib here to compute this in parallel?
    for method_name in method_names:
        attrs = batch_attr[method_name].cpu().numpy()
        if segmented:
            assert isinstance(ds, SegSensNDataset)
            attrs = segment_attributions(
                ds.segmented_images.cpu().numpy(), attrs
            )
        # [batch_size, 1, -1]
        attrs = attrs.reshape((attrs.shape[0], 1, -1))
        for n_idx, n in enumerate(ds.n_range):
            # [batch_size, num_subsets, n]
            n_mask_attrs = np.take_along_axis(
                attrs, axis=-1, indices=removed_indices[n]
            )
            for activation_fn in activation_fns:
                # Compute sum of attributions
                # [batch_size, num_subsets]
                n_sum_of_attrs = n_mask_attrs.sum(axis=-1)
                n_output_diffs = output_diffs[activation_fn][n]
                # Compute correlation between output difference and
                # sum of attribution values
                result[activation_fn][method_name][:, n_idx] = torch.tensor(
                    rowwise_pearsonr(n_sum_of_attrs, n_output_diffs)
                )
    return result


def _sens_n_batch(
    samples: torch.Tensor,
    labels: torch.Tensor,
    model: Callable,
    attrs: Dict[str, torch.Tensor],
    maskers: Mapping[str, Masker],
    activation_fns: List[str],
    n_range: npt.NDArray,
    num_subsets: int,
    segmented: bool,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    method_names = list(attrs.keys())
    orig_output = _get_orig_output(samples, model, activation_fns)
    # masker_name -> activation_fn -> method_name -> [batch_size, num_steps]
    batch_result: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    for masker_name, masker in maskers.items():
        # Create pseudo-dataset to generate perturbed samples
        if segmented:
            ds = SegSensNDataset(n_range, num_subsets, samples)
            assert isinstance(masker, ImageMasker)
            ds.set_masker(masker)
        else:
            ds = SensitivityNDataset(n_range, num_subsets, samples, masker)

        output_diffs, removed_indices = _compute_out_diffs(
            model, ds, activation_fns, orig_output, labels
        )

        batch_result[masker_name] = _compute_correlations(
            method_names,
            attrs,
            ds,
            segmented,
            removed_indices,
            output_diffs,
            activation_fns,
        )
    return batch_result


def sensitivity_n(
    model: nn.Module,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    maskers: Mapping[str, Masker],
    activation_fns: str | List[str],
    min_subset_size: float,
    max_subset_size: float,
    num_steps: int,
    num_subsets: int,
    segmented: bool,
    device: torch.device = torch.device("cpu"),
) -> SensitivityNResult:
    """Computes the Sensitivity-n metric for a given :class:`~attribench.data.AttributionsDataset` and model.

    Sensitivity-n is computed by iteratively masking a random subset of `n` features
    of the input samples and computing the output of the model on the masked
    samples.

    For each random subset of masked features, the sum of the attributions is
    also computed. This results in two series of values: the model output and
    the sum of the attributions. The Sensitivity-n metric is the correlation
    between these two series.

    This is repeated for different values of `n` between `min_subset_size` and
    `max_subset_size` in `num_steps` steps. `min_subset_size` and `max_subset_size`
    are percentages of the total number of features.
    For each value of `n`, `num_subsets` random subsets are generated.

    If segmented is True, then the Seg-Sensitivity-n metric is computed.
    This metric is analogous to Sensitivity-n, but instead of using random
    subsets of features, the images are first segmented into superpixels and
    then random subsets of superpixels are masked. This improves the
    signal-to-noise ratio of the metric for high-resolution images.

    The Sensitivity-n metric is computed for each masker in `maskers` and for each
    activation function in `activation_fns`.

    Parameters
    ----------
    model : nn.Module
        Model to compute Sensitivity-n for.
    attributions_dataset : AttributionsDataset
        Dataset containing the attributions to compute Sensitivity-n on.
    batch_size : int
        Batch size to use when computing model output on masked samples.
    maskers : Dict[str, Masker]
        Dictionary of maskers to use. Keys are the names of the maskers.
    activation_fns : Union[Tuple[str], str]
        Activation functions to use. If a single string is passed, then the
        it is converted to a single-element list.
    min_subset_size : float
        Minimum percentage of features to mask.
    max_subset_size : float
        Maximum percentage of features to mask.
    num_steps : int
        Number of steps between `min_subset_size` and `max_subset_size`.
    num_subsets : int
        Number of random subsets to generate for each value of `n`.
    segmented : bool
        If True, then the Seg-Sensitivity-n metric is computed.
    device : torch.device, optional
        Device to use, by default torch.device("cpu")

    Returns
    -------
    SensitivityNResult
    """
    if isinstance(activation_fns, str):
        activation_fns = [activation_fns]

    model.to(device)
    model.eval()

    grouped_dataset = GroupedAttributionsDataset(attributions_dataset)
    dataloader = DataLoader(
        grouped_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    result = SensitivityNResult(
        attributions_dataset.method_names,
        list(maskers.keys()),
        list(activation_fns),
        num_samples=attributions_dataset.num_samples,
        num_steps=num_steps,
    )

    # Compute range of subset sizes
    n_range = np.linspace(min_subset_size, max_subset_size, num_steps)
    if segmented:
        n_range = n_range * 100
    else:
        total_num_features = np.prod(attributions_dataset.attributions_shape)
        n_range = n_range * total_num_features
    n_range = n_range.astype(int)

    for (
        batch_indices,
        batch_x,
        batch_y,
        batch_attr,
    ) in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_result = _sens_n_batch(
            batch_x,
            batch_y,
            model,
            batch_attr,
            maskers,
            activation_fns,
            n_range,
            num_subsets,
            segmented,
        )
        result.add(GroupedBatchResult(batch_indices, batch_result))
    return result

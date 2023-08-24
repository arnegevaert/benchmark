from attribench.result import MaxSensitivityResult
from attribench.result._grouped_batch_result import GroupedBatchResult
from typing import Dict
from attribench.data import AttributionsDataset
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
)
import torch
from attribench._attribution_method import AttributionMethod
from torch.utils.data import DataLoader
import math
from tqdm import tqdm


def _normalize_attrs(attrs):
    flattened = attrs.flatten(1)
    denom = torch.norm(flattened, dim=1, p=math.inf, keepdim=True) + 1e-8
    return flattened / denom


def _max_sensitivity_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_attr: Dict[str, torch.Tensor],
    method_dict: Dict[str, AttributionMethod],
    num_perturbations: int,
    radius: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if set(method_dict.keys()) != set(batch_attr.keys()):
        print(method_dict.keys())
        print(batch_attr.keys())
        raise ValueError(
            "Method dictionary and batch attributions dictionary"
            " must have the same keys."
        )
    result: Dict[str, torch.Tensor] = {
        method_name: torch.zeros(1) for method_name in method_dict.keys()
    }
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # Compute Max-Sensitivity for each method
    for method_name, method in method_dict.items():
        #attrs = _normalize_attrs(batch_attr[method_name])
        # TODO we are not taking into account the aggregation here
        # TODO also the AttributionsDataset is not being used
        attrs = _normalize_attrs(method(batch_x, batch_y).detach()).cpu()
        diffs = []

        for _ in range(num_perturbations):
            # Add uniform noise with infinity norm <= radius
            # torch.rand generates noise between 0 and 1
            # => This generates noise between -radius and radius
            noise = (
                torch.rand(batch_x.shape, device=device) * 2 * radius - radius
            )
            noisy_samples = batch_x + noise
            # Get new attributions from noisy samples
            noisy_attrs = _normalize_attrs(
                method(noisy_samples, batch_y).detach()
            )
            # Get relative norm of attribution difference
            # [batch_size]
            diffs.append(torch.norm(noisy_attrs.cpu() - attrs, dim=1).detach())
        # [batch_size, num_perturbations]
        diffs = torch.stack(diffs, 1)
        # [batch_size]
        result[method_name] = diffs.max(dim=1)[0].cpu()
    return result


def max_sensitivity(
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    method_dict: Dict[str, AttributionMethod],
    num_perturbations: int,
    radius: float,
    device: torch.device = torch.device("cpu"),
) -> MaxSensitivityResult:
    """Computes the Max-Sensitivity metric for a given `Dataset` and attribution
    methods. Max-Sensitivity is computed by adding a small amount of uniform noise
    to the input samples and computing the norm of the difference in attributions
    between the original samples and the noisy samples.
    The maximum norm of difference is then taken as the Max-Sensitivity.

    The idea is that a small amount of noise should not change the attributions
    significantly, so the norm of the difference should be small. If the norm
    is large, then the attributions are not robust to small perturbations in the
    input.

    Parameters
    ----------
    attributions_dataset : Dataset
        The dataset for which the Max-Sensitivity should be computed.
    batch_size : int
        The batch size to use for computing the Max-Sensitivity.
    method_dict : Dict[str, AttributionMethod]
        Dictionary of attribution methods for which the Max-Sensitivity should be
        computed.
    num_perturbations : int
        The number of perturbations to use for computing the Max-Sensitivity.
    radius : float
        The radius of the uniform noise to add to the input samples.
    device : torch.device, optional
        Device to use, by default `torch.device("cpu")`.
    """
    grouped_dataset = GroupedAttributionsDataset(attributions_dataset)
    dataloader = DataLoader(
        grouped_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    method_names = list(method_dict.keys())
    result = MaxSensitivityResult(
        method_names,
        num_samples=len(grouped_dataset),
    )
    for batch_indices, batch_x, batch_y, batch_attr in tqdm(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_result = _max_sensitivity_batch(
            batch_x,
            batch_y,
            batch_attr,
            method_dict,
            num_perturbations,
            radius,
            device,
        )
        result.add(GroupedBatchResult(batch_indices, batch_result))
    return result
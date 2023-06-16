from attribench.result import MaxSensitivityResult
from attribench.result._batch_result import BatchResult
from typing import Dict
from torch.utils.data import Dataset
import torch
from attribench._attribution_method import AttributionMethod
from torch.utils.data import DataLoader
from attribench.data import IndexDataset
import math


def _normalize_attrs(attrs):
    flattened = attrs.flatten(1)
    return flattened / torch.norm(flattened, dim=1, p=math.inf, keepdim=True)


def max_sensitivity_batch(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    method_dict: Dict[str, AttributionMethod],
    num_perturbations: int,
    radius: float,
    device: torch.device,
):
    result: Dict[str, torch.Tensor] = {
        method_name: torch.zeros(1) for method_name in method_dict.keys()
    }
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # Compute Max-Sensitivity for each method
    for method_name, method in method_dict.items():
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
    dataset: Dataset,
    method_dict: Dict[str, AttributionMethod],
    batch_size: int,
    num_perturbations: int,
    radius: float,
    device: torch.device = torch.device("cpu"),
):
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
    dataset : Dataset
        The dataset for which the Max-Sensitivity should be computed.
    method_dict : Dict[str, AttributionMethod]
        Dictionary of attribution methods for which the Max-Sensitivity should be
        computed.
    batch_size : int
        The batch size to use for computing the Max-Sensitivity.
    num_perturbations : int
        The number of perturbations to use for computing the Max-Sensitivity.
    radius : float
        The radius of the uniform noise to add to the input samples.
    device : torch.device, optional
        Device to use, by default `torch.device("cpu")`.
    """
    index_dataset = IndexDataset(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    method_names = tuple(method_dict.keys())
    result = MaxSensitivityResult(
        method_names,
        shape=(len(index_dataset),),
    )
    for batch_indices, batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_result = max_sensitivity_batch(
            batch_x, batch_y, method_dict, num_perturbations, radius, device
        )
        result.add(
            BatchResult(batch_indices, batch_result, list(method_names))
        )

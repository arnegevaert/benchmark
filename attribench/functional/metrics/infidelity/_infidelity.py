from tqdm import tqdm
from torch import nn
import torch
from typing import Dict, List
from attribench.data.attributions_dataset._attributions_dataset import (
    GroupedAttributionsDataset,
    AttributionsDataset,
)
from ._perturbation_generator import PerturbationGenerator
from torch.utils.data import DataLoader
from attribench._activation_fns import ACTIVATION_FNS
from attribench.result._infidelity_result import InfidelityResult
from attribench.result._grouped_batch_result import GroupedBatchResult


def _infidelity_batch(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_attr: Dict[str, torch.Tensor],
    perturbation_generators: Dict[str, PerturbationGenerator],
    num_perturbations: int,
    activation_fns: List[str],
    device: torch.device,
):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    method_names: List[str] = list(batch_attr.keys())

    # method_name -> perturbation_generator
    # -> activation_fn -> [batch_size, 1]
    batch_result: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
        method_name: {
            pg_name: {
                activation_fn: torch.zeros(1)
                for activation_fn in activation_fns
            }
            for pg_name in perturbation_generators.keys()
        }
        for method_name in method_names
    }

    # Attributions need to be moved to the device,
    # because we will be computing dot products between
    # attributions and perturbations later.
    # They also need to have the same shape as the samples for this.
    # Any axis that has length 1 in the attributions is repeated to
    # match the sample shape
    tensor_attributions: Dict[str, torch.Tensor] = {}
    for attribution_method, attributions in batch_attr.items():
        for axis in range(len(attributions.shape)):
            if attributions.shape[axis] == 1:
                attributions = torch.repeat_interleave(
                    attributions, batch_x.shape[axis], dim=axis
                )
        tensor_attributions[attribution_method] = attributions.flatten(1).to(
            device
        )

    # Get original model output on the samples
    # (dict: activation_fn -> torch.Tensor)
    orig_output = {}
    with torch.no_grad():
        for fn in activation_fns:
            # [batch_size, 1]
            orig_output[fn] = ACTIVATION_FNS[fn](model(batch_x)).gather(
                dim=1, index=batch_y.unsqueeze(-1)
            )

    for (
        pert_name,
        pert_generator,
    ) in perturbation_generators.items():
        pert_generator.set_samples(batch_x)
        dot_products: Dict[str, List[torch.Tensor]] = {
            method: [] for method in batch_attr.keys()
        }
        pred_diffs = {afn: [] for afn in activation_fns}

        for _ in range(num_perturbations):
            # Get perturbation vector I and perturbed samples (x - I)
            perturbation_vector = pert_generator.generate_perturbation()
            perturbed_x = batch_x - perturbation_vector

            # Get output of model on perturbed sample
            with torch.no_grad():
                perturbed_output = model(perturbed_x)

            # Save the prediction difference and perturbation vector
            for fn in activation_fns:
                activated_perturbed_output = ACTIVATION_FNS[fn](
                    perturbed_output
                ).gather(dim=1, index=batch_y.unsqueeze(-1))
                pred_diffs[fn].append(
                    torch.squeeze(orig_output[fn] - activated_perturbed_output)
                )

            # Compute dot products of perturbation vectors with all
            # attributions for each sample
            for (
                attribution_method,
                attributions,
            ) in tensor_attributions.items():
                # (batch_size)
                dot_products[attribution_method].append(
                    (perturbation_vector.flatten(1) * attributions).sum(dim=-1)
                )

        # For each method and activation function, compute infidelity
        # activation_fn -> [num_perturbations, batch_size]
        tensor_pred_diffs = {
            afn: torch.stack(pred_diffs[afn], dim=0)
            for afn in pred_diffs.keys()
        }
        for method in tensor_attributions.keys():
            # Dot prodcts for this method
            method_dot_products = torch.stack(
                dot_products[method]
            )  # [num_perturbations, batch_size]
            # Denominator for normalizing constant beta
            beta_denominator = torch.mean(
                method_dot_products**2, dim=0, keepdim=True
            )  # [1, batch_size]
            for afn in activation_fns:
                # Numerator for normalizing constant beta depends on
                # activation function
                # [1, batch_size]
                beta_numerator = torch.mean(
                    method_dot_products * tensor_pred_diffs[afn],
                    dim=0,
                    keepdim=True,
                )
                beta = beta_numerator / beta_denominator
                # If attribution map is constant 0,
                # dot products will be 0 and beta will be nan or inf. Set to 0.
                beta[torch.isnan(beta)] = 0
                beta[torch.isinf(beta)] = 0
                # [batch_size, 1]
                infidelity = torch.mean(
                    (beta * method_dot_products - tensor_pred_diffs[afn]) ** 2,
                    dim=0,
                ).unsqueeze(-1)
                batch_result[method][pert_name][afn] = (
                    infidelity.cpu().detach().numpy()
                )
    return batch_result


def infidelity(
    model: nn.Module,
    attributions_dataset: AttributionsDataset,
    batch_size: int,
    activation_fns: List[str],
    perturbation_generators: Dict[str, PerturbationGenerator],
    num_perturbations: int,
    device: torch.device = torch.device("cpu"),
) -> InfidelityResult:
    """Computes the Infidelity metric for a given :class:`~attribench.data.AttributionsDataset` and model.

    Infidelity is computed by generating perturbations for each sample in the
    dataset and computing the difference in the model's output on the original
    sample and the perturbed sample. This difference is then compared to the
    dot product of the perturbation vector and the attribution map for each
    attribution method. The Infidelity metric is the mean squared error between
    these two values.

    The idea is that if the dot product is large, then the perturbation vector
    is aligned with the attribution map, and the model's output should change
    significantly when the perturbation is applied. If the dot product is small,
    then the perturbation vector is not aligned with the attribution map, and
    the model's output should not change significantly when the perturbation is
    applied.

    The mean squared error is computed for `num_perturbations` perturbations
    for each sample. The `perturbation_generators` argument is a dictionary
    mapping perturbation generator names to `PerturbationGenerator` objects.
    These objects can be used to implement different versions of Infidelity.

    The Infidelity metric is computed for each perturbation generator in
    `perturbation_generators` and each activation function in `activation_fns`.

    Parameters
    ----------
    model : nn.Module
        Model to compute Infidelity on.
    attributions_dataset : AttributionsDataset
        Dataset of attributions to compute Infidelity on.
    batch_size : int
        Batch size to use when computing Infidelity.
    perturbation_generators : Dict[str, PerturbationGenerator]
        Dictionary of perturbation generators to use for generating
        perturbations.
    num_perturbations : int
        Number of perturbations to generate for each sample.
    activation_fns : Tuple[str]
        Tuple of activation functions to use when computing Infidelity.
    device : torch.device, optional
        Device to use, by default `torch.device("cpu")`
    """
    grouped_dataset = GroupedAttributionsDataset(attributions_dataset)
    dataloader = DataLoader(
        grouped_dataset, batch_size=batch_size, num_workers=4
    )
    result = InfidelityResult(
        attributions_dataset.method_names,
        list(perturbation_generators.keys()),
        activation_fns,
        num_samples=attributions_dataset.num_samples,
    )
    for batch_indices, batch_x, batch_y, batch_attr in tqdm(dataloader):
        batch_result = _infidelity_batch(
            model,
            batch_x,
            batch_y,
            batch_attr,
            perturbation_generators,
            num_perturbations,
            activation_fns,
            device,
        )
        result.add(GroupedBatchResult(batch_indices, batch_result))
    return result

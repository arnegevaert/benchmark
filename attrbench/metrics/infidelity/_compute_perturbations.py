from typing import Callable, Tuple, Dict

import torch
import numpy as np

from attrbench.lib import AttributionWriter
from attrbench.lib.util import ACTIVATION_FNS
from .perturbation_generator import PerturbationGenerator


def _mse(a, b):
    return ((a - b) ** 2).mean(axis=1, keepdims=True)


def _compute_perturbations(samples: torch.Tensor, labels: torch.Tensor, model: Callable,
                           attrs: Dict[str, np.ndarray],
                           perturbation_generator: PerturbationGenerator,
                           num_perturbations: int,
                           activation_fns: Tuple[str],
                           writer: AttributionWriter = None) -> Dict[str, Dict[str, np.ndarray]]:
    # Move attributions to samples device
    tensor_attrs: Dict[str, torch.tensor] = {}
    for key, value in attrs.items():
        if value.shape[1] != samples.shape[1]:
            value = np.repeat(value, samples.shape[1], axis=1)
        tensor_attrs[key] = torch.tensor(value, device=samples.device).flatten(1)

    # Get original model output
    orig_output = {}
    with torch.no_grad():
        for fn in activation_fns:
            orig_output[fn] = ACTIVATION_FNS[fn](model(samples)).gather(dim=1,
                                                                        index=labels.unsqueeze(-1))  # [batch_size, 1]

    perturbation_generator.set_samples(samples)

    dot_products: Dict[str, list] = {m_name: [] for m_name in tensor_attrs}
    pred_diffs: Dict[str, list] = {afn: [] for afn in activation_fns}
    for i_pert in range(num_perturbations):
        # Get perturbation vector I and perturbed samples (x - I)
        perturbation_vector = perturbation_generator()
        perturbed_samples = samples - perturbation_vector
        if writer:
            writer.add_images("perturbation_vector", perturbation_vector, global_step=i_pert)
            writer.add_images("perturbed_samples", perturbed_samples, global_step=i_pert)

        # Get output of model on perturbed sample
        with torch.no_grad():
            perturbed_output = model(perturbed_samples)
        # Save the prediction difference and perturbation vector
        for fn in activation_fns:
            act_pert_out = ACTIVATION_FNS[fn](perturbed_output).gather(dim=1, index=labels.unsqueeze(-1))
            # [batch_size]
            pred_diffs[fn].append(torch.squeeze(orig_output[fn] - act_pert_out))

        # Compute dot products of perturbation vectors with all attributions for each sample
        for key, value in tensor_attrs.items():
            # [batch_size]
            dot_products[key].append((perturbation_vector.flatten(1) * value).sum(dim=-1))

    # For each method and activation function, compute infidelity
    # Result: activation_function -> method_name -> [batch_size, 1]
    result: Dict[str, Dict[str, np.ndarray]] = {afn: {m_name: None for m_name in tensor_attrs} for afn in activation_fns}
    # activation_function -> [num_perturbations, batch_size]
    tensor_pred_diffs = {afn: torch.stack(pred_diffs[afn], dim=0) for afn in pred_diffs.keys()}
    for m_name in tensor_attrs.keys():
        # Dot products for this method
        m_dot_products = torch.stack(dot_products[m_name])  # [num_perturbations, batch_size]
        # Denominator for normalizing constant beta
        beta_den = torch.mean(m_dot_products**2, dim=0, keepdim=True)  # [1, batch_size]
        for afn in activation_fns:
            # Numerator for normalizing constant beta depends on activation function
            beta_num = torch.mean(m_dot_products * tensor_pred_diffs[afn], dim=0, keepdim=True)  # [1, batch_size]
            beta = beta_num / beta_den  # [1, batch_size]
            # If attribution map is constant 0, dot products will be 0 and beta will be nan. Set to 0.
            beta[torch.isnan(beta)] = 0
            infid = torch.mean((beta * m_dot_products - tensor_pred_diffs[afn])**2, dim=0).unsqueeze(-1)
            result[afn][m_name] = infid.cpu().detach().numpy()
    return result

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
    t_attrs: Dict[str, torch.tensor] = {}
    for key, value in attrs.items():
        if value.shape[1] != samples.shape[1]:
            value = np.repeat(value, samples.shape[1], axis=1)
        t_attrs[key] = torch.tensor(value, device=samples.device).flatten(1)

    # Get original model output
    orig_output = {}
    with torch.no_grad():
        for fn in activation_fns:
            orig_output[fn] = ACTIVATION_FNS[fn](model(samples)).gather(dim=1,
                                                                        index=labels.unsqueeze(-1))  # [batch_size, 1]

    perturbation_generator.set_samples(samples)
    squared_errors: Dict[str, Dict[str, list]] = {afn: {m_name: [] for m_name in t_attrs} for afn in activation_fns}
    for i_pert in range(num_perturbations):
        pred_diffs = {}
        dot_products = {}

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
            pred_diffs[fn] = orig_output[fn] - act_pert_out

        # Compute dot products of perturbation vectors with all attributions for each sample
        for key, value in t_attrs.items():
            # [batch_size]
            dot_products[key] = (perturbation_vector.flatten(1) * value).sum(dim=-1, keepdim=True)

        # Compute squared error for each combination of activation function and method
        for afn in activation_fns:
            for m_name, dot_product in dot_products.items():
                squared_errors[afn][m_name].append((dot_product - pred_diffs[afn])**2)

    result = {afn: {} for afn in activation_fns}
    for afn in activation_fns:
        for m_name, sq_errors in squared_errors[afn].items():
            result[afn][m_name] = torch.cat(sq_errors, dim=1).mean(dim=1, keepdim=True).cpu().detach().numpy()
    return result

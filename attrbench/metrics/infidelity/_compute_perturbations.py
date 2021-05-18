from typing import Callable, Tuple, Dict

import torch
import numpy as np

from attrbench.lib import AttributionWriter
from attrbench.lib.util import ACTIVATION_FNS
from .perturbation_generator import PerturbationGenerator


def _compute_perturbations(samples: torch.Tensor, labels: torch.Tensor, model: Callable,
                           perturbation_generator: PerturbationGenerator,
                           num_perturbations: int,
                           activation_fns: Tuple[str],
                           writer: AttributionWriter = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Get original model output
    orig_output = {}
    with torch.no_grad():
        for fn in activation_fns:
            orig_output[fn] = ACTIVATION_FNS[fn](model(samples)).gather(dim=1,
                                                                        index=labels.unsqueeze(-1))  # [batch_size, 1]

    pert_vectors = []
    pred_diffs: Dict[str, list] = {fn: [] for fn in activation_fns}
    perturbation_generator.set_samples(samples)
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
            pred_diffs[fn].append(orig_output[fn] - act_pert_out)
        pert_vectors.append(perturbation_vector.cpu())  # [batch_size, *sample_shape]
    pert_vectors = torch.stack(pert_vectors, dim=1).numpy()  # [batch_size, num_perturbations, *sample_shape]
    res_pred_diffs: Dict[str, np.ndarray] = {}
    for fn in activation_fns:
        res_pred_diffs[fn] = torch.cat(pred_diffs[fn], dim=1).cpu().numpy()  # [batch_size, num_perturbations]
    return pert_vectors, res_pred_diffs

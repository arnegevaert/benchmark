from typing import Callable, Tuple, Dict

import torch
from torch.utils.data import DataLoader

from attrbench.lib import AttributionWriter
from attrbench.lib.util import ACTIVATION_FNS
from ._dataset import _GaussianPerturbation, _SegmentRemovalPerturbation, _SquareRemovalPerturbation

_PERTURBATION_CLASSES = {
    "gaussian": _GaussianPerturbation,
    "square": _SquareRemovalPerturbation,
    "segment": _SegmentRemovalPerturbation
}


def _compute_perturbations(samples: torch.Tensor, labels: torch.Tensor, model: Callable, perturbation_mode: str,
                           perturbation_size: float, num_perturbations: int, activation_fn: Tuple[str],
                           writer: AttributionWriter = None, num_workers=0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = samples.device
    if perturbation_mode not in _PERTURBATION_CLASSES.keys():
        raise ValueError(f"Invalid perturbation mode {perturbation_mode}. "
                         f"Valid options are {', '.join(list(_PERTURBATION_CLASSES.keys()))}")
    perturbation_ds = _PERTURBATION_CLASSES[perturbation_mode](samples.cpu().numpy(),
                                                               perturbation_size, num_perturbations)
    perturbation_dl = DataLoader(perturbation_ds, batch_size=1, num_workers=num_workers)

    # Get original model output
    orig_output = {}
    with torch.no_grad():
        for fn in activation_fn:
            orig_output[fn] = ACTIVATION_FNS[fn](model(samples)).gather(dim=1,
                                                                        index=labels.unsqueeze(-1))  # [batch_size, 1]

    pert_vectors = []
    pred_diffs: Dict[str, list] = {fn: [] for fn in activation_fn}
    for i_pert, (perturbed_samples, perturbation_vector) in enumerate(perturbation_dl):
        # Get perturbation vector I and perturbed samples (x - I)
        perturbed_samples = perturbed_samples[0].float().to(device)
        perturbation_vector = perturbation_vector[0].float()
        if writer:
            writer.add_images("perturbation_vector", perturbation_vector, global_step=i_pert)
            writer.add_images("perturbed_samples", perturbed_samples, global_step=i_pert)

        # Get output of model on perturbed sample
        with torch.no_grad():
            perturbed_output = model(perturbed_samples)
        # Save the prediction difference and perturbation vector
        for fn in activation_fn:
            act_pert_out = ACTIVATION_FNS[fn](perturbed_output).gather(dim=1, index=labels.unsqueeze(-1))
            pred_diffs[fn].append(orig_output[fn] - act_pert_out)
        pert_vectors.append(perturbation_vector)  # [batch_size, *sample_shape]
    pert_vectors = torch.stack(pert_vectors, dim=1)  # [batch_size, num_perturbations, *sample_shape]
    res_pred_diffs: Dict[str, torch.Tensor] = {}
    for fn in activation_fn:
        res_pred_diffs[fn] = torch.cat(pred_diffs[fn], dim=1).cpu()  # [batch_size, num_perturbations]
    return pert_vectors, res_pred_diffs

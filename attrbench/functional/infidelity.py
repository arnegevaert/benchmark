import torch
from typing import Callable
import random


def _gaussian_perturbation(samples, stdev):
    """
    Calculates and applies Gaussian perturbation
    :param samples: Input samples
    :param stdev: Standard deviation
    :return: (perturbed samples, perturbation vector (I in paper))
    """
    perturbation_vector = torch.randn(samples.shape, device=samples.device) * stdev
    perturbed_samples = samples - perturbation_vector
    return perturbed_samples, perturbation_vector


def _square_removal_perturbation(samples, square_size):
    height = samples.shape[2]
    width = samples.shape[3]
    square_size_int = int(square_size * height)
    x_loc = random.randint(0, width - square_size_int)
    y_loc = random.randint(0, height - square_size_int)
    perturbation_mask = torch.zeros(samples.shape)
    perturbation_mask[:, :, x_loc:x_loc + square_size_int, y_loc:y_loc+square_size_int] = 1
    perturbation_vector = samples * perturbation_mask.to(samples.device)
    perturbed_samples = samples - perturbation_vector
    return perturbed_samples, perturbation_vector


def _baseline_subset_perturbation(samples, subset_size):
    pass


_PERTURBATION_FUNCTIONS = {
    "gaussian": _gaussian_perturbation,
    "square": _square_removal_perturbation,
    "baseline_subset": _baseline_subset_perturbation
}


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
               perturbation_mode: str, perturbation_size: float, num_perturbations: int,
               writer=None):
    if perturbation_mode not in _PERTURBATION_FUNCTIONS.keys():
        raise ValueError(f"Invalid perturbation mode {perturbation_mode}. "
                         f"Valid options are {', '.join(list(_PERTURBATION_FUNCTIONS.keys()))}")
    perturbation_fn = _PERTURBATION_FUNCTIONS[perturbation_mode]

    # Get original model output
    with torch.no_grad():
        orig_output = (model(samples)).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]
    attrs = method(samples, labels).detach()

    # Replicate attributions along channel dimension if necessary (if explanation has fewer channels than image)
    if attrs.shape[1] != samples.shape[1]:
        shape = [1 for _ in range(len(attrs.shape))]
        shape[1] = samples.shape[1]
        attrs = attrs.repeat(*tuple(shape))
    attrs_flattened = attrs.flatten(1)

    infid = []
    for i_pert in range(num_perturbations):
        # Get perturbation vector I and perturbed samples (x - I)
        perturbed_samples, perturbation_vector = perturbation_fn(samples, perturbation_size)
        if writer:
            writer.add_images("perturbation_vector", perturbation_vector, global_step=i_pert)
            writer.add_images("perturbed_samples", perturbed_samples, global_step=i_pert)

        with torch.no_grad():
            perturbed_output = model(perturbed_samples).gather(dim=1, index=labels.unsqueeze(-1))
        # Calculate dot product between each sample and its corresponding perturbation vector
        # This is equivalent to diagonal of matmul
        dot_product = (attrs_flattened * perturbation_vector.flatten(1)).sum(dim=1, keepdim=True)  # [batch_size, 1]
        pred_diff = orig_output - perturbed_output
        infid.append(((dot_product - pred_diff)**2).detach())
    # Take average across all perturbations
    infid = torch.cat(infid, dim=1).mean(dim=1)  # [batch_size]
    infid = infid.unsqueeze(1)  # [batch_size, 1]

    if writer:
        writer.add_images("samples", samples)
        writer.add_images("attrs", attrs)

    return infid.cpu()

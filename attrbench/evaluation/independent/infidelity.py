import torch
from typing import Callable, List


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
               perturbation_range: List[float], num_perturbations: int, debug_mode=False):
    result = []
    device = samples.device
    n_channels = samples.size(1)
    # Get original model output
    with torch.no_grad():
        orig_output = (model(samples)).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]
    attrs = method(samples, labels)
    if attrs.shape != samples.shape:
        raise ValueError("Attributions must have same shape as samples for infidelity")

    debug_data = []
    for eps in perturbation_range:
        eps_result = []
        if debug_mode:
            debug_data.append({
                "perturbations": [],
                "perturbed_samples": []
            })
        for _ in range(num_perturbations):
            perturbation = torch.randn(samples.shape, device=device) * eps
            perturbed = samples + perturbation  # [batch_size, n_channels, width, height]
            if debug_mode:
                debug_data[-1]["perturbations"].append(perturbation)
                debug_data[-1]["perturbed_samples"].append(perturbed)
            with torch.no_grad():
                perturbed_output = model(perturbed).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]
            # We calculate X * I (p3 in paper).
            # perturbation is \eps.
            # I = x - z_0, where z_0 = x_0 + \eps.
            # I = x - z_0 = x_0 - x_0 - \eps = -\eps.
            # I = -\eps = -perturbation
            # Note: element-wise product followed by sum(dim=1) == diagonal of matmul (without doing entire matmul)
            dot_product = (attrs.flatten(1) * -perturbation.flatten(1)).sum(dim=1, keepdim=True)  # [batch_size, 1]
            eps_result.append((dot_product - (orig_output - perturbed_output))**2)  # [batch_size, 1]
        result.append(torch.stack(eps_result, dim=1).mean(dim=1))  # [batch_size]
    result = torch.cat(result, dim=1).cpu().detach()  # [batch_size, len(perturbation_range)]
    if debug_mode:
        debug_result = {
            "attrs": attrs,
            "pert_data": debug_data
        }
        return result, debug_result
    return result

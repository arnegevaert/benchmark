import torch
from typing import Callable, List


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
               perturbation_range: List[float], num_perturbations: int):
    result = []
    device = samples.device
    n_channels = samples.size(1)
    # Get original model output
    with torch.no_grad():
        orig_output = (model(samples)).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]

    for eps in perturbation_range:
        perturbation = torch.randn(samples.shape, device=device) * eps
        perturbed = samples + perturbation  # [batch_size, n_channels, width, height]
        with torch.no_grad():
            perturbed_output = model(perturbed).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]
        eps_result = []
        explanation = method(samples, labels)
        # If explanation is on pixel level, we need to replicate value for each pixel n_channels times,
        # since current_perturbation is [batch_size, n_channels, width, height]
        if explanation.size(1) == 1:
            explanation = explanation.repeat(1, n_channels, 1, 1)
        for _ in range(num_perturbations):
            # We calculate X * I (p3 in paper).
            # perturbation is \eps.
            # I = x - z_0, where z_0 = x_0 + \eps.
            # I = x - z_0 = x_0 - x_0 - \eps = -\eps.
            # I = -\eps = -perturbation
            dot_product = (explanation.flatten(1) * -perturbation.flatten(1)).sum(dim=1, keepdim=True)  # [batch_size, 1]
            eps_result.append((dot_product - (orig_output - perturbed_output))**2)  # [batch_size, 1]
        result.append(torch.stack(eps_result, dim=1).mean(dim=1))  # [batch_size]
    return torch.cat(result, dim=1)  # [batch_size, len(perturbation_range)]

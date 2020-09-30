from typing import Callable, List
import torch


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, perturbation_range: List[float],
                    num_perturbations: int):
    result = []
    device = samples.device

    attrs = method(samples, labels)  # [batch_size, *sample_shape]
    norm = torch.norm(attrs.flatten(1), dim=1)
    for eps in perturbation_range:
        max_diff = 0
        for _ in range(num_perturbations):
            # Add uniform noise in [-eps, eps]
            noise = torch.rand(samples.shape, device=device) * 2 * eps - eps
            noisy_samples = samples + noise
            # Get new attributions from noisy samples
            noisy_attrs = method(noisy_samples, labels)
            # Get relative norm of attribution difference
            diffs = torch.norm(noisy_attrs.flatten(1) - attrs.flatten(1), dim=1) / norm
            max_diff = max(max_diff, diffs.max().cpu().detach().item())
        result.append(max_diff)
    # [len(perturbation_range)]
    return torch.tensor(result)

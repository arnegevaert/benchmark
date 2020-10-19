from typing import Callable, List
import torch


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, perturbation_range: List[float],
                    num_perturbations: int):
    result = []
    device = samples.device

    attrs = method(samples, labels)  # [batch_size, *sample_shape]
    norm = torch.norm(attrs.flatten(1), dim=1)
    for eps in perturbation_range:
        diffs = []
        for _ in range(num_perturbations):
            # Add uniform noise in [-eps, eps]
            noise = torch.rand(samples.shape, device=device) * 2 * eps - eps
            noisy_samples = samples + noise
            # Get new attributions from noisy samples
            noisy_attrs = method(noisy_samples, labels)
            # Get relative norm of attribution difference
            # [batch_size]
            diffs.append((torch.norm(noisy_attrs.flatten(1) - attrs.flatten(1), dim=1) / norm).detach())
        # [batch_size, num_perturbations]
        diffs = torch.stack(diffs, 1)
        # [batch_size]
        result.append(diffs.max(dim=1)[0])
    # [batch_size, len(perturbation_range)]
    return torch.stack(result, dim=1).cpu()

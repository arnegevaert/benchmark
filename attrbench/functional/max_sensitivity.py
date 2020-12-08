from typing import Callable, List
import torch


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, perturbation_range: List[float],
                    num_perturbations: int, debug_mode=False):
    result = []
    device = samples.device

    attrs = method(samples, labels).detach()  # [batch_size, *sample_shape]
    norm = torch.norm(attrs.flatten(1), dim=1)
    debug_data = []
    for eps in perturbation_range:
        diffs = []
        if debug_mode:
            debug_data.append({
                "perturbations": [],
                "perturbed_samples": []
            })
        for _ in range(num_perturbations):
            # Add uniform noise in [-eps, eps]
            noise = torch.rand(samples.shape, device=device) * 2 * eps - eps
            noisy_samples = samples + noise
            if debug_mode:
                debug_data[-1]["perturbations"].append(noise)
                debug_data[-1]["perturbed_samples"].append(noisy_samples)
            # Get new attributions from noisy samples
            noisy_attrs = method(noisy_samples, labels).detach()
            # Get relative norm of attribution difference
            # [batch_size]
            diffs.append((torch.norm(noisy_attrs.flatten(1) - attrs.flatten(1), dim=1) / norm).detach())
        # [batch_size, num_perturbations]
        diffs = torch.stack(diffs, 1)
        # [batch_size]
        result.append(diffs.max(dim=1)[0])
    # [batch_size, len(perturbation_range)]
    result = torch.stack(result, dim=1).cpu()
    if debug_mode:
        debug_result = {
            "attrs": attrs,
            "pert_data": debug_data
        }
        return result, debug_result
    return result

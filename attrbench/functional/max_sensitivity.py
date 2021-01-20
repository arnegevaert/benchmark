from typing import Callable, List
import torch


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, perturbation_range: List[float],
                    num_perturbations: int,attrs, debug_mode=False,writer=None):
    result = []
    device = samples.device
    if attrs is None:
        attrs = method(samples, labels).detach()  # [batch_size, *sample_shape]
    attrs = attrs.to(device)
    norm = torch.norm(attrs.flatten(1), dim=1)
    debug_data = []
    for eps in perturbation_range:
        diffs = []
        if debug_mode:
            debug_data.append({
                "perturbations": [],
                "perturbed_samples": []
            })
        for n_p in range(num_perturbations):
            # Add uniform noise in [-eps, eps]
            noise = torch.rand(samples.shape, device=device) * 2 * eps - eps
            noisy_samples = samples + noise
            if debug_mode:
                writer.add_images('perturbations eps: {}'.format(eps), noise, global_step=n_p)
                writer.add_images('perturbed_samples eps: {}'.format(eps), noisy_samples, global_step=n_p)
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
        writer.add_images('attributions', attrs)

    return result

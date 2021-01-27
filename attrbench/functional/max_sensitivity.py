from typing import Callable
import torch
import math


def _normalize_attrs(attrs):
    flattened = attrs.flatten(1)
    return flattened / torch.norm(flattened, dim=1, p=math.inf, keepdim=True)


def max_sensitivity(samples: torch.Tensor, labels: torch.Tensor, method: Callable, attrs: torch.Tensor, radius: float,
                    num_perturbations: int, writer=None):
    device = samples.device
    if attrs is None:
        attrs = method(samples, labels).detach()  # [batch_size, *sample_shape]
    attrs = _normalize_attrs(attrs)  # [batch_size, -1]

    diffs = []
    for n_p in range(num_perturbations):
        # Add uniform noise with infinity norm <= radius
        # torch.rand generates noise between 0 and 1 => This generates noise between -radius and radius
        noise = torch.rand(samples.shape, device=device) * 2 * radius - radius
        noisy_samples = samples + noise
        if writer:
            writer.add_images('perturbations', noise, global_step=n_p)
            writer.add_images('perturbed_samples', noisy_samples, global_step=n_p)
        # Get new attributions from noisy samples
        noisy_attrs = _normalize_attrs(method(noisy_samples, labels).detach())
        # Get relative norm of attribution difference
        # [batch_size]
        diffs.append(torch.norm(noisy_attrs - attrs, dim=1).detach())
    # [batch_size, num_perturbations]
    diffs = torch.stack(diffs, 1)
    # [batch_size]
    result = diffs.max(dim=1, keepdim=True)[0].cpu()

    if writer:
        writer.add_images('attributions', attrs)

    return result

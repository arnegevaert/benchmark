import torch
from typing import Callable, List


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
               perturbation_range: List[float], num_perturbations: int, debug_mode: bool=False,
               writer=None):
    result = []
    device = samples.device
    # Get original model output
    with torch.no_grad():
        orig_output = (model(samples)).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]
    if attrs is None:
        attrs = method(samples, labels).detach()
    attrs=attrs.to(device)
    # Replicate attributions along channel dimension if necessary
    if attrs.shape[1] != samples.shape[1]:
        shape = [1 for _ in range(len(attrs.shape))]
        shape[1] = samples.shape[1]
        attrs = attrs.repeat(*tuple(shape))


    for eps in perturbation_range:
        eps_result = []

        for n_p in range(num_perturbations):
            perturbation = torch.randn(samples.shape, device=device) * eps
            perturbed = samples + perturbation  # [batch_size, n_channels, width, height]
            if debug_mode:
                writer.add_images('perturbations eps: {}'.format(eps), perturbation, global_step=n_p)
                writer.add_images('perturbed_samples eps: {}'.format(eps), perturbed, global_step=n_p)


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
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)
    return result

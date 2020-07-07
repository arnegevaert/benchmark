import torch
import numpy as np
from typing import Iterable, Callable, Dict
from tqdm import tqdm


# TODO support different perturbations (currently hardcoded Gaussian noise with sigma=0.2, same as in paper source)
def infidelity(data: Iterable, model: Callable, methods: Dict[str, Callable],
               n_perturbations: int, pixel_level: bool, device: str):
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        batch_result = {m_name: [] for m_name in methods}
        samples = samples.to(device)
        labels = labels.to(device)
        n_channels = samples.shape[1]
        # Get original model output
        orig_output = model(samples).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()  # [batch_size]
        tiled = samples.repeat(n_perturbations, 1, 1, 1)  # [batch_size*n_perturbations, n_channels, width, height]
        tiled = tiled.view(n_perturbations, *samples.shape)  # [n_perturbations, batch_size, n_channels, width, height]
        # TODO for some reason, results from paper can only be reproduced if perturbations are clamped like this
        perturbation = torch.clamp(torch.randn(tiled.shape, device=device) * 0.2, min=-1.)
        perturbed = tiled - perturbation  # [n_perturbations, batch_size, n_channels, width, height]
        #perturbation = torch.randn(tiled.shape) * 0.2
        #perturbed = tiled + perturbation
        for perturbation_idx in range(n_perturbations):
            # current_perturbation: [batch_size, n_channels*width*height]
            current_perturbation = perturbation[perturbation_idx].flatten(start_dim=1)
            current_perturbed = perturbed[perturbation_idx]  # [batch_size, n_channels, width, height]
            perturbed_output = model(current_perturbed).gather(dim=1, index=labels.unsqueeze(-1)).squeeze()  # [batch_size]
            for m_name in methods:
                explanation = methods[m_name](samples, labels)
                # If explanation is on pixel level, we need to replicate value for each pixel n_channels times,
                # since current_perturbation is [batch_size, n_channels, width, height]
                if pixel_level:
                    explanation = explanation.unsqueeze(1).repeat(1, n_channels, 1, 1)
                explanation_flattened = torch.flatten(explanation, start_dim=1)  # [batch_size, n_channels*width*height]
                # calculate dot product between perturbation and explanation for each sample in batch
                dot_product = (explanation_flattened * current_perturbation).sum(dim=1)  # [batch_size]
                sample_infidelity = (dot_product - (orig_output - perturbed_output))**2
                batch_result[m_name].append(sample_infidelity)
                #result[m_name].append(sample_infidelity.cpu().detach().numpy())
        for m_name in batch_result:
            for sample_infidelity in result[m_name]:
                result[m_name].append(sample_infidelity.cpu().detach().numpy())
    return {m_name: np.concatenate(result[m_name]) for m_name in methods}

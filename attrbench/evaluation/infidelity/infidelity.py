import torch
from typing import Iterable, Callable, Dict, List
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult
from attrbench.evaluation.util import transform_fns


def infidelity(data: Iterable, model: Callable, methods: Dict[str, Callable],
               perturbation_range: List[float], n_perturbations: int,
               pixel_level: bool, device: str, output_transform: str):
    result = {m_name: [[] for _ in perturbation_range] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        n_channels = samples.shape[1]
        # Get original model output
        with torch.no_grad():
            orig_output = transform_fns[output_transform](model(samples))\
                .gather(dim=1, index=labels.unsqueeze(-1)).squeeze()  # [batch_size]
        tiled = samples.repeat(n_perturbations, 1, 1, 1)  # [batch_size*n_perturbations, n_channels, width, height]
        tiled = tiled.view(n_perturbations, *samples.shape)  # [n_perturbations, batch_size, n_channels, width, height]

        for pert_idx, pert_value in enumerate(perturbation_range):
            # We assume samples are normalized to have mean=0 and sdev=1, so perturbations above 1 are too high
            perturbation = torch.randn(tiled.shape, device=device) * pert_value
            perturbed = tiled + perturbation  # [n_perturbations, batch_size, n_channels, width, height]
            with torch.no_grad():
                model_output = transform_fns[output_transform](model(perturbed.flatten(0, 1)))\
                    .view(n_perturbations, samples.size(0), -1)
            for m_name in methods:
                m_result = []
                explanation = methods[m_name](samples, labels)
                # If explanation is on pixel level, we need to replicate value for each pixel n_channels times,
                # since current_perturbation is [batch_size, n_channels, width, height]
                if pixel_level:
                    explanation = explanation.unsqueeze(1).repeat(1, n_channels, 1, 1)
                explanation_flattened = explanation.flatten(1)  # [batch_size, n_channels*width*height]
                for perturbation_idx in range(n_perturbations):
                    # current_perturbation: [batch_size, n_channels*width*height]
                    # Reason for taking negative perturbation: current_perturbation is vector I in paper (p3).
                    # Here defined as: I = x - z_0, where z_0 = x_0 + \eps.
                    # In other words: I = x - z_0 = x_0 - x_0 - \eps = -\eps.
                    # \eps is perturbation, so here we use -perturbation
                    current_perturbation = -perturbation[perturbation_idx].flatten(start_dim=1)
                    perturbed_output = model_output[perturbation_idx].gather(dim=1, index=labels.unsqueeze(-1)).squeeze()  # [batch_size]
                    # calculate dot product between perturbation and explanation for each sample in batch
                    dot_product = (explanation_flattened * current_perturbation).sum(dim=1)  # [batch_size]
                    m_result.append((dot_product - (orig_output - perturbed_output))**2)
                batch_infidelity = torch.stack(m_result, dim=0).mean(dim=0)  # [batch_size]
                result[m_name][pert_idx].append(batch_infidelity.cpu().detach())
    for m_name in methods:
        for pert_idx in range(len(perturbation_range)):
            # [n_batches*batch_size]
            result[m_name][pert_idx] = torch.cat(result[m_name][pert_idx])
        # [n_batches*batch_size, len(pert_range)]
        result[m_name] = torch.stack(result[m_name], dim=0).numpy().transpose()
    return LinePlotResult(data=result, x_range=perturbation_range)

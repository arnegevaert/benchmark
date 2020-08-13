from typing import Iterable, Callable, List, Dict
import torch
import numpy as np
import math
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult


def max_sensitivity(data: Iterable, methods: Dict[str, Callable],
                    perturbation_range: List[float],
                    device: str):
    result = {m_name: [-math.inf for _ in perturbation_range] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        for m_name in methods:
            attrs = methods[m_name](samples, labels)  # [batch_size, *sample_shape]
            norm = torch.norm(attrs.flatten(1), dim=1)
            for eps_i, eps in enumerate(perturbation_range):
                # Uniform noise in [-eps, eps]
                noise = torch.rand(samples.shape, device=device) * 2 * eps - eps
                noisy_samples = samples + noise
                noisy_attrs = methods[m_name](noisy_samples, labels)
                diffs = torch.norm(noisy_attrs.flatten(1) - attrs.flatten(1), dim=1) / norm
                result[m_name][eps_i] = max(result[m_name][eps_i], diffs.max().cpu().detach().item())
    result_data = {
        m_name: np.array(result[m_name]).reshape(1, -1) for m_name in methods
    }
    return LinePlotResult(result_data, perturbation_range)

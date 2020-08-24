import numpy as np
from typing import Iterable, List, Dict, Callable
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult
from attrbench.evaluation.util import mask_pixels
import torch


def impact_score(data: Iterable, model: Callable, mask_range: List[int], methods: Dict[str, Callable],
                 mask_value: float, tau: float, device: str = "cpu"):
    strict_score_counts = {m_name: [0 for _ in mask_range] for m_name in methods}
    i_score_counts = {m_name: [0 for _ in mask_range] for m_name in methods}
    total_samples = 0

    for b, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            orig_out = model(samples)
        # Only consider samples that were correctly classified
        # TODO consider also using misclassified samples (but then use model output instead of labels)
        y_pred = torch.argmax(orig_out, dim=1)
        reduced_batch_size = (y_pred == labels).sum().item()  # Amount of correctly classified samples
        total_samples += reduced_batch_size
        samples = samples[y_pred == labels]
        orig_out = orig_out[y_pred == labels]
        labels = labels[y_pred == labels]
        orig_out = orig_out[torch.arange(reduced_batch_size), labels]  # [reduced_batch_size, 1]
        for m_name in methods:
            # Get and sort original attributions
            attrs = methods[m_name](samples, target=labels)
            pixel_level_attrs = len(attrs.shape) == 3
            attrs = attrs.flatten(1)
            sorted_indices = attrs.argsort().cpu()
            for n_idx, n in enumerate(mask_range):
                masked_samples = mask_pixels(samples, sorted_indices[:, -n:], mask_value, pixel_level_attrs)
                with torch.no_grad():
                    masked_out = model(masked_samples)
                predictions = torch.argmax(masked_out, dim=1)
                prediction_flipped = predictions != labels
                confidence_values = masked_out[torch.arange(reduced_batch_size), labels]
                confidence_dropped = confidence_values <= orig_out * tau
                # Calculate counts for (strict) i_score
                # We divide by total number of samples in the last step
                # I_{strict} = 1/n * sum_n (y_i' \neq y_i)
                strict_score_counts[m_name][n_idx] += prediction_flipped.sum().item()
                # I = 1/n * sum_n ((y_i' \neq y_i) \vee (z_i' \leq z_i)
                i_score_counts[m_name][n_idx] += (prediction_flipped | confidence_dropped).sum().item()

    i_score_data = {
        m_name: np.array([i_score_counts[m_name]]) / total_samples for m_name in methods
    }
    strict_score_data = {
        m_name: np.array([strict_score_counts[m_name]]) / total_samples for m_name in methods
    }
    return LinePlotResult(data=i_score_data, x_range=mask_range), LinePlotResult(data=strict_score_data, x_range=mask_range)

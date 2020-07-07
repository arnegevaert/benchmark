from typing import Iterable, Callable, List, Dict
import numpy as np
import torch
from tqdm import tqdm


# TODO duplicated code in deletion curves
def insertion_curves(data: Iterable, model: Callable, methods: Dict[str, Callable],
                     insert_range: List[int], background_value: float, pixel_level_mask: bool,
                     device: str):
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        for key in methods:
            batch_result = []
            attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
            # Flatten each sample in order to sort indices per sample
            attrs = attrs.flatten(1)  # [batch_size, -1]
            # Sort indices of attrs in ascending order
            sorted_indices = attrs.argsort().cpu().detach().numpy()
            for i in insert_range:
                # Get indices of i most important inputs
                to_insert = sorted_indices[:, -i:]  # [batch_size, i]
                inserted_samples = torch.ones(samples.shape).to(device) * background_value
                batch_dim = np.column_stack([range(samples.shape[0]) for _ in range(i)])
                if pixel_level_mask:
                    unraveled = np.unravel_index(to_insert, samples.shape[2:])
                    inserted_samples[(batch_dim, -1, *unraveled)] = samples[(batch_dim, -1, *unraveled)]
                else:
                    unraveled = np.unravel_index(to_insert, samples.shape[1:])
                    inserted_samples[(batch_dim, *unraveled)] = samples[(batch_dim, *unraveled)]
                # Get predictions for result
                predictions = model(inserted_samples)
                predictions = predictions[np.arange(predictions.shape[0]), labels].reshape(-1, 1)
                batch_result.append(predictions.cpu().detach().numpy())
            batch_result = np.concatenate(batch_result, axis=1)
            result[key].append(batch_result)
    # [n_batches*batch_size, len(mask_range)]
    return {m_name: np.concatenate(result[m_name], axis=0) for m_name in methods}

from typing import Iterable, Callable, List, Dict
import numpy as np


def simple_sensitivity(data: Iterable, model: Callable, methods: Dict[str, Callable],
                       mask_range: List[int], mask_value: float):
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(data):
        for key in methods:
            batch_result = []
            attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
            # Flatten each sample in order to sort indices per sample
            attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
            # Sort indices of attrs in ascending order
            sorted_indices = attrs.argsort()
            for i in mask_range:
                # Get indices of i most important inputs
                to_mask = sorted_indices[:, -i:]  # [batch-size, i]
                unraveled = np.unravel_index(to_mask, samples.shape[1:])
                # Mask i most important inputs
                # batch_dim: [batch_size, i] (made to match unravel_index output)
                batch_size = samples.shape[0]
                batch_dim = np.array(list(range(batch_size))*i).reshape(-1, batch_size).transpose()
                masked_samples = samples.clone()
                masked_samples[(batch_dim, *unraveled)] = mask_value
                # Get predictions for result
                predictions = model(masked_samples)
                predictions = predictions[np.arange(predictions.shape[0]), labels].reshape(-1, 1)
                batch_result.append(predictions.detach().numpy())
            batch_result = np.concatenate(batch_result, axis=1)
            result[key].append(batch_result)
    return {m_name: np.concatenate(result[m_name], axis=0) for m_name in methods}  # [n_batches*batch_size, len(mask_range)]

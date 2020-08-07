from typing import Iterable, Callable, List, Dict
import numpy as np
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult
import torch


def insertion_deletion_curves(data: Iterable, model: Callable, methods: Dict[str, Callable],
                              mask_range: List[int], mask_value: float, pixel_level_mask: bool,
                              device: str, mode: str):
    assert mode in ["deletion", "insertion"]
    result = {m_name: [] for m_name in methods}
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        # Check which samples are classified correctly
        # Only want to calculate for correctly classified samples
        with torch.no_grad():
            y_pred = torch.argmax(model(samples), dim=1)
        samples = samples[y_pred == labels]
        labels = labels[y_pred == labels]
        for key in methods:
            batch_result = []
            attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
            # Flatten each sample in order to sort indices per sample
            attrs = attrs.flatten(1)  # [batch_size, -1]
            # Sort indices of attrs in ascending order
            sorted_indices = attrs.argsort().cpu().detach().numpy()
            # TODO create all masked samples at once for more efficient GPU usage
            for i in mask_range:
                # Get indices of i most important inputs
                masked_samples = samples.clone() if mode == "deletion" \
                    else torch.ones(samples.shape).to(device) * mask_value
                if i > 0:
                    to_mask = sorted_indices[:, -i:]  # [batch-size, i]
                    batch_dim = np.column_stack([range(samples.shape[0]) for _ in range(i)])
                    if pixel_level_mask:
                        unraveled = np.unravel_index(to_mask, samples.shape[2:])
                        masked_samples[(batch_dim, -1, *unraveled)] = mask_value if mode == "deletion" \
                            else samples[(batch_dim, -1, *unraveled)]
                    else:
                        unraveled = np.unravel_index(to_mask, samples.shape[1:])
                        masked_samples[(batch_dim, *unraveled)] = mask_value if mode == "deletion" \
                            else samples[(batch_dim, *unraveled)]
                # Get predictions for result
                predictions = model(masked_samples)
                predictions = predictions[np.arange(predictions.shape[0]), labels].reshape(-1, 1)
                batch_result.append(predictions.cpu().detach().numpy())
            batch_result = np.concatenate(batch_result, axis=1)
            result[key].append(batch_result)
    data = {
        # [n_batches*batch_size, len(mask_range)]
        m_name: np.concatenate(result[m_name], axis=0) for m_name in methods
    }
    return Result(data, mask_range, mode)


class Result(LinePlotResult):
    def __init__(self, data, x_range, mode):
        super().__init__(data, x_range)
        normalized = {}
        for method in self.processed:
            for key in ["mean", "lower", "upper"]:
                arr = self.processed[method][key]
                if mode == "deletion":
                    normalized[method][key] = (arr - arr[-1]) / (arr[0] - arr[-1])
                else:
                    normalized[method][key] = (arr - arr[0]) / (arr[-1] - arr[0])
        self.processed = normalized

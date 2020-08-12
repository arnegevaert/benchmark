from typing import Iterable, Callable, List, Dict
import numpy as np
from tqdm import tqdm
from attrbench.evaluation.result import LinePlotResult
from attrbench.evaluation.util import transform_fns
import torch


def _mask_pixels(imgs, indices, mask_value, pixel_level_mask):
    batch_size, color_channels = imgs.shape[:2]
    num_pixels = indices.shape[1]
    result = imgs.clone()
    batch_dim = np.column_stack([range(batch_size) for _ in range(num_pixels)])
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        for color_dim in range(color_channels):
            result[(batch_dim, color_dim, *unraveled)] = mask_value
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = mask_value
    return result


def _insert_pixels(imgs, indices, mask_value, pixel_level_mask):
    num_pixels = indices.shape[1]
    batch_size, color_channels = imgs.shape[:2]
    result = torch.ones(imgs.shape) * mask_value
    batch_dim = np.column_stack([range(batch_size) for _ in range(num_pixels)])
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        for color_dim in range(color_channels):
            result[(batch_dim, color_dim, *unraveled)] = imgs[(batch_dim, color_dim, *unraveled)]
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = imgs[(batch_dim, *unraveled)]
    return result


def insertion_deletion_curves(data: Iterable, sample_shape, model: Callable, methods: Dict[str, Callable],
                              mask_range: List[int], mask_value: float, pixel_level_mask: bool,
                              device: str, mode: str, output_transform: str):
    assert output_transform in ["identity", "softmax", "logit_softmax"]
    assert mode in ["deletion", "insertion"]
    result = {m_name: [] for m_name in methods}
    full_mask_output = model(torch.ones((1, *sample_shape)).to(device) * mask_value)
    full_mask_output = transform_fns[output_transform](full_mask_output)
    for batch_index, (samples, labels) in enumerate(tqdm(data)):
        samples = samples.to(device)
        labels = labels.to(device)
        # Check which samples are classified correctly
        # Only want to calculate for correctly classified samples
        with torch.no_grad():
            y_pred = torch.argmax(model(samples), dim=1)
        samples = samples[y_pred == labels]
        labels = labels[y_pred == labels]
        batch_size = samples.size(0)
        if batch_size > 0:
            orig_predictions = transform_fns[output_transform](model(samples))
            orig_predictions = orig_predictions[np.arange(batch_size), labels].reshape(-1, 1)
            full_mask_predictions = full_mask_output[[0]*batch_size, labels].reshape(-1, 1)
            for key in methods:
                batch_result = []
                attrs = methods[key](samples, labels)  # [batch_size, *sample_shape]
                # Flatten each sample in order to sort indices per sample
                attrs = attrs.flatten(1)  # [batch_size, -1]
                # Sort indices of attrs in ascending order
                sorted_indices = attrs.argsort().cpu().detach().numpy()

                for i in mask_range:
                    # Mask/insert pixels
                    if mode == "deletion":
                        masked_samples = _mask_pixels(samples, sorted_indices[:, -i:],
                                                      mask_value, pixel_level_mask)
                    else:
                        masked_samples = _insert_pixels(samples, sorted_indices[:, -i:],
                                                        mask_value, pixel_level_mask)
                    # Get predictions for result
                    predictions = transform_fns[output_transform](model(masked_samples))
                    predictions = predictions[np.arange(batch_size), labels].reshape(-1, 1)
                    batch_result.append(predictions.cpu().detach().numpy())

                if mode == "deletion":
                    batch_result = np.concatenate([
                        orig_predictions.cpu().detach().numpy(),
                        np.concatenate(batch_result, axis=1),
                        full_mask_predictions.cpu().detach().numpy()
                    ], axis=1)
                else:
                    batch_result = np.concatenate([
                        full_mask_predictions.cpu().detach().numpy(),
                        np.concatenate(batch_result, axis=1),
                        orig_predictions.cpu().detach().numpy()
                    ], axis=1)

                result[key].append(batch_result)
    res_data = {
        # [n_batches*batch_size, len(mask_range) + 2]
        m_name: np.concatenate(result[m_name], axis=0) for m_name in methods
    }
    mask_range = [0] + mask_range + \
                 [(np.product(sample_shape[-2:]) if pixel_level_mask else np.product(sample_shape[-3:]))]
    return LinePlotResult(res_data, mask_range)

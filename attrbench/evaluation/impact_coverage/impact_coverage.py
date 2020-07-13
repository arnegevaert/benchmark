import torch
import numpy as np
from sklearn.metrics import jaccard_score
import random
from typing import Iterable, Callable, Dict, Tuple
from tqdm import tqdm


def impact_coverage(data: Iterable, patch, target_label: int,
                    model: Callable, methods: Dict[str, Callable], device: str = "cpu"):
    # image_size = sample_shape[-1]
    patch_size = patch.shape[-1]
    keep_list = []  # boolean mask of images to keep (not predicted as target class and successfully attacked)
    patch_masks = []  # boolean mask of location of patch
    critical_factor_mask = {m_name: [] for m_name in methods}  # boolean mask of location top factors

    for b, (samples, labels) in enumerate(tqdm(data)):
        image_size = samples.shape[-1]
        samples = samples.to(device, non_blocking=True)
        # Get model predictions on original samples
        predictions = model(samples).cpu()
        # Apply patch to all images in batch (random location, but same for each image in batch)
        ind = random.randint(0, image_size - patch_size)
        samples[:, :, ind:ind + patch_size, ind:ind + patch_size] = patch
        # Save mask with pixels covered by patch
        patch_location_mask = np.zeros(samples.shape)
        patch_location_mask[:, :, ind:ind + patch_size, ind: ind + patch_size] = 1.
        patch_masks.append(patch_location_mask)
        adv_out = model(samples).cpu()
        # use images that are not of the targeted class and are successfully attacked
        keep_indices = (predictions.argmax(axis=1) != target_label) * (adv_out.argmax(axis=1) == target_label) * (labels != target_label)
        keep_list.extend(keep_indices)
        for m_name in critical_factor_mask:
            method = methods[m_name]
            attrs = method(samples, target=target_label)
            flattened_attrs = attrs.reshape(attrs.shape[0], -1)
            sorted_indices = flattened_attrs.argsort().cpu()
            masks = np.zeros(samples.shape)
            if len(attrs.shape) == 4:
                # Attributions are per color channel
                nr_top_attributions = attrs.shape[1] * np.prod(patch.shape).item()
                to_mask = sorted_indices[:, -nr_top_attributions:]  # [batch_size, i]
                batch_dim = np.column_stack([range(samples.shape[0]) for _ in range(nr_top_attributions)])
                unraveled = np.unravel_index(to_mask, samples.shape[1:])
                masks[(batch_dim, *unraveled)] = 1.
            elif len(attrs.shape) == 3:
                # Attributions are per pixel
                nr_top_attributions = np.prod(patch.shape).item()
                to_mask = sorted_indices[:, -nr_top_attributions:]  # [batch_size, i]
                batch_dim = np.column_stack([range(samples.shape[0]) for _ in range(nr_top_attributions)])
                unraveled = np.unravel_index(to_mask, samples.shape[2:])
                masks[(batch_dim, -1, *unraveled)] = 1.
            else:
                raise ValueError("Attributions must have 3 (per-pixel) or 4 (per-channel) dimensions."
                                 f"Shape was f{attrs.shape}")
            critical_factor_mask[m_name].append(masks)
    result = {}
    patch_masks = np.vstack(patch_masks)[keep_list]
    for m_name in critical_factor_mask:
        cr_f_m = np.vstack(critical_factor_mask[m_name])
        cr_f_m = cr_f_m[keep_list]
        result[m_name] = jaccard_score(patch_masks.flatten(),cr_f_m.flatten())
    return result

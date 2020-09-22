import numpy as np
from typing import Iterable, Callable, Dict
from tqdm import tqdm
from result import BoxPlotResult


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
        # indx = random.randint(0, image_size - patch_size)
        # indy = random.randint(0, image_size - patch_size)
        indx = image_size // 2 - patch_size // 2  # place patch in center
        indy = indx
        samples[:, :, indx:indx + patch_size, indy:indy + patch_size] = patch
        # Save mask with pixels covered by patch

        for m_name in critical_factor_mask:
            method = methods[m_name]
            attrs = method(samples, target=target_label)
            flattened_attrs = attrs.reshape(attrs.shape[0], -1)
            sorted_indices = flattened_attrs.argsort().cpu()
            masks = np.zeros(attrs.shape)
            if len(attrs.shape) == 4:
                # Attributions are per color channel
                nr_top_attributions = attrs.shape[1] * np.prod(patch.shape[2:]).item()
            elif len(attrs.shape) == 3:
                # Attributions are per pixel
                nr_top_attributions = np.prod(patch.shape[2:]).item()
            else:
                raise ValueError("Attributions must have 3 (per-pixel) or 4 (per-channel) dimensions."
                                 f"Shape was f{attrs.shape}")
            to_mask = sorted_indices[:, -nr_top_attributions:]  # [batch_size, i]
            batch_dim = np.tile(range(samples.shape[0]), (nr_top_attributions, 1)).transpose()
            unraveled = np.unravel_index(to_mask, attrs.shape[1:])
            masks[(batch_dim, *unraveled)] = 1.
            critical_factor_mask[m_name].append(masks)

        patch_location_mask = np.zeros(attrs.shape)
        if len(attrs.shape) == 3:  # Attributions are per pixel location
            patch_location_mask[:, indx:indx + patch_size, indy: indy + patch_size] = 1
        else:  # Attributions are per color channel
            patch_location_mask[:, :, indx:indx + patch_size, indy: indy + patch_size] = 1
        patch_masks.append(patch_location_mask)
        adv_out = model(samples).cpu()
        # keep only images that are not of the targeted class and are successfully attacked
        keep_indices = (predictions.argmax(axis=1) != target_label) * (adv_out.argmax(axis=1) == target_label) * (labels != target_label)
        keep_list.extend(keep_indices)
    res_data = {}
    patch_masks = np.vstack(patch_masks)[keep_list] # locations of mask in the images,
    for m_name in critical_factor_mask:
        cr_f_m = np.vstack(critical_factor_mask[m_name])
        cr_f_m = cr_f_m[keep_list]
        patch_masks_flattened = patch_masks.reshape(patch_masks.shape[0], -1).astype(np.bool)
        cr_f_m_flattened = cr_f_m.reshape(cr_f_m.shape[0], -1).astype(np.bool)
        intersect = (patch_masks_flattened & cr_f_m_flattened).sum(axis=1)
        union = (patch_masks_flattened | cr_f_m_flattened).sum(axis=1)
        res_data[m_name] = intersect / union
    return BoxPlotResult(res_data)

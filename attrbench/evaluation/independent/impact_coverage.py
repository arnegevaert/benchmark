from typing import Callable
from attrbench.lib import PixelMaskingPolicy
import torch


def impact_coverage(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                    patch: torch.Tensor, target_label: int):
    if len(samples.shape) != 4:
           raise ValueError("Impact Coverage can only be computed for image data and expects 4 input dimensions")
    image_size = samples.shape[-1]
    patch_size = patch.shape[-1]
    orig_out = model(samples)

    # Apply patch to all images in batch (random location, but same for each image in batch)
    # indx = random.randint(0, image_size - patch_size)
    # indy = random.randint(0, image_size - patch_size)
    indx = image_size // 2 - patch_size // 2  # place patch in center
    indy = indx
    samples[:, :, indx:indx + patch_size, indy:indy + patch_size] = patch

    # Create masks from top n attributions
    attrs = method(samples, target=target_label)
    if attrs.shape[1] != 1:
        raise ValueError(f"Impact Coverage expects per-pixel attributions (shape of attrs must have 1 color channel). Actual shape was {attrs.shape}.")
    flattened_attrs = attrs.flatten(1)
    sorted_indices = flattened_attrs.argsort().cpu()
    nr_top_attributions = patch_size**2

    to_mask = sorted_indices[:, -nr_top_attributions:]
    pmp = PixelMaskingPolicy(mask_value=1.)
    critical_factor_mask = pmp(torch.zeros(attrs.shape), to_mask)

    # Create masks from patch itself
    patch_mask = torch.zeros(attrs.shape)
    patch_mask[:, :, indx:indx + patch_size, indy:indy + patch_size] = 1

    # Get model output on adversarial sample
    adv_out = model(samples)
    # Determine which images were not originally of targeted class and successfully attacked
    keep = (orig_out.argmax(axis=1) != target_label) & \
           (adv_out.argmax(axis=1) == target_label) & \
           (labels != target_label)

    # Calculate IoU of critical factors (top n attributions) with adversarial patch
    patch_mask_flattened = patch_mask.flatten(1).bool()
    critical_factor_mask_flattened = critical_factor_mask.flatten(1).bool()
    intersection = (patch_mask_flattened & critical_factor_mask_flattened).sum(dim=1)
    union = (patch_mask_flattened | critical_factor_mask_flattened).sum(dim=1)
    # [batch_size], [batch_size]
    return intersection.float() / union.float(), keep.cpu()

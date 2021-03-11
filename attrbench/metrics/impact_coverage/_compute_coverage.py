from typing import Callable

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib.masking import ConstantMasker


def _compute_coverage(attacked_samples: torch.Tensor, method: Callable, patch_mask: torch.Tensor,
                      targets: torch.Tensor, writer: AttributionWriter = None) -> torch.Tensor:
    # Get attributions
    attrs = method(attacked_samples, target=targets).detach()
    # Check attributions shape
    if attrs.shape[1] not in (1, 3):
        raise ValueError(f"Impact Coverage only works on image data. Attributions must have 1 or 3 color channels."
                         f"Found attributions shape {attrs.shape}.")
    # If attributions have only 1 color channel, we need a single-channel patch mask as well
    if attrs.shape[1] == 1:
        patch_mask = patch_mask[:, 0, :, :]
    # Get indices of top k attributions
    flattened_attrs = attrs.flatten(1)
    sorted_indices = flattened_attrs.argsort().cpu()
    # Number of top attributions is equal to number of features masked by the patch
    # We assume here that the mask is the same size for all samples!
    nr_top_attributions = patch_mask[0, ...].long().sum().item()

    # Create mask of critical factors (most important pixels/features according to attributions)
    to_mask = sorted_indices[:, -nr_top_attributions:]
    # TODO don't use a masker for this
    masker = ConstantMasker(feature_level="pixel" if attrs.shape[1] == 1 else "channel", mask_value=1.)
    # Initialize as constant zeros, "mask" the most important features with 1
    critical_factor_mask = np.zeros(attrs.shape)
    masker.initialize_baselines(critical_factor_mask)
    critical_factor_mask = masker.mask(critical_factor_mask, to_mask)
    critical_factor_mask = critical_factor_mask.reshape(critical_factor_mask.shape[0], -1).astype(np.bool)

    # Calculate IoU of critical factors (top n attributions) with adversarial patch
    patch_mask_flattened = patch_mask.flatten(1).bool().numpy()
    intersection = (patch_mask_flattened & critical_factor_mask).sum(axis=1)
    union = (patch_mask_flattened | critical_factor_mask).sum(axis=1)
    iou = intersection.astype(np.float) / union.astype(np.float)
    if writer:
        writer.add_images('Attacked samples', attacked_samples)
        writer.add_images('Attacked attributions', attrs)
    # [batch_size]
    return torch.tensor(iou)
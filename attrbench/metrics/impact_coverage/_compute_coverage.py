from typing import Callable

import numpy as np
import torch

from attrbench.lib import AttributionWriter


def _compute_coverage(attacked_samples: torch.Tensor, patch_mask: torch.Tensor, targets: torch.Tensor, method: Callable = None,
                      attrs: np.ndarray = None,
                      writer: AttributionWriter = None) -> torch.Tensor:
    if method is None and attrs is None:
        raise ValueError("Specify an attribution method or attributions array")
    # Get attributions
    if method is not None:
        attrs = method(attacked_samples, target=targets).detach().cpu().numpy()
    # Check attributions shape
    if attrs.shape[1] not in (1, 3):
        raise ValueError(f"Impact Coverage only works on image data. Attributions must have 1 or 3 color channels."
                         f"Found attributions shape {attrs.shape}.")
    # If attributions have only 1 color channel, we need a single-channel patch mask as well
    if attrs.shape[1] == 1:
        patch_mask = patch_mask[:, 0, :, :]
    # Get indices of top k attributions
    flattened_attrs = attrs.reshape(attrs.shape[0], -1)
    sorted_indices = flattened_attrs.argsort()
    # Number of top attributions is equal to number of features masked by the patch
    # We assume here that the mask is the same size for all samples!
    nr_top_attributions = patch_mask[0, ...].long().sum().item()

    # Create mask of critical factors (most important pixels/features according to attributions)
    to_mask = sorted_indices[:, -nr_top_attributions:]
    critical_factor_mask = np.zeros(attrs.shape).reshape(attrs.shape[0], -1)
    batch_size = attrs.shape[0]
    batch_dim = np.tile(range(batch_size), (nr_top_attributions, 1)).transpose()
    critical_factor_mask[batch_dim, to_mask] = 1
    critical_factor_mask = critical_factor_mask.astype(np.bool)

    # Calculate IoU of critical factors (top n attributions) with adversarial patch
    patch_mask_flattened = patch_mask.flatten(1).bool().numpy()
    intersection = (patch_mask_flattened & critical_factor_mask).sum(axis=1)
    union = (patch_mask_flattened | critical_factor_mask).sum(axis=1)
    iou = intersection.astype(np.float) / union.astype(np.float)
    if writer:
        writer.add_images("Attacked samples", attacked_samples)
        writer.add_attribution('Attacked attributions', attrs)
    # [batch_size]
    return torch.tensor(iou)

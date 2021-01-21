from typing import Callable
from attrbench.lib import PixelMaskingPolicy, FeatureMaskingPolicy
import random
import torch
from os import path


def impact_coverage(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                    patch_folder: str, debug_mode=False,writer=None):
    # TODO this is temporary, will be fixed when fixing issue #86
    if not path.isfile(path.join(patch_folder, "resnet18.pt")):
        raise ValueError(f"File not found: {path.join(patch_folder, 'resnet18.pt')}")
    patch = torch.load(path.join(patch_folder, "resnet18.pt"))
    target_label = 0
    if len(samples.shape) != 4:
           raise ValueError("Impact Coverage can only be computed for image data and expects 4 input dimensions")
    samples = samples.clone()
    image_size = samples.shape[-1]
    patch_size = patch.shape[-1]
    original_output = model(samples)

    # Apply patch to all images in batch (random location, but same for each image in batch)
    indx = random.randint(0, image_size - patch_size)
    indy = random.randint(0, image_size - patch_size)
    # place patch in center
    #indx = image_size // 2 - patch_size // 2
    #indy = indx
    samples[:, :, indx:indx + patch_size, indy:indy + patch_size] = patch

    # Get attributions
    attrs = method(samples, target=target_label).detach()
    # Check attributions shape
    if attrs.shape[1] not in (1, 3):
        raise ValueError(f"Impact Coverage only works on image data. Attributions must have 1 or 3 color channels."
                         f"Found attributions shape {attrs.shape}.")
    # Get indices of top k attributions
    flattened_attrs = attrs.flatten(1)
    sorted_indices = flattened_attrs.argsort().cpu()
    # Number of top attributions is equal to number of features masked by the patch
    # If attributions are pixel level, this is the size of the patch
    # If attributions are channel level, this is the size of the patch * the number of channels (3)
    nr_top_attributions = patch_size**2 * attrs.shape[1]

    # Create mask of critical factors (most important pixels/features according to attributions)
    to_mask = sorted_indices[:, -nr_top_attributions:]
    pmp = PixelMaskingPolicy(mask_value=1.) if attrs.shape[1] == 1 else FeatureMaskingPolicy(mask_value=1.)
    critical_factor_mask = pmp(torch.zeros(attrs.shape), to_mask)

    # Create masks from patch itself
    patch_mask = torch.zeros(attrs.shape)
    patch_mask[:, :, indx:indx + patch_size, indy:indy + patch_size] = 1

    # Get model output on adversarial sample
    adv_out = model(samples)
    # Determine which images were not originally of targeted class and successfully attacked
    keep = (original_output.argmax(axis=1) != target_label) & \
           (adv_out.argmax(axis=1) == target_label) & \
           (labels != target_label)

    # Calculate IoU of critical factors (top n attributions) with adversarial patch
    patch_mask_flattened = patch_mask.flatten(1).bool()
    critical_factor_mask_flattened = critical_factor_mask.flatten(1).bool()
    intersection = (patch_mask_flattened & critical_factor_mask_flattened).sum(dim=1)
    union = (patch_mask_flattened | critical_factor_mask_flattened).sum(dim=1)
    iou = intersection.float() / union.float()
    if debug_mode:
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)
    # [batch_size], [batch_size]
    return iou, keep.cpu()

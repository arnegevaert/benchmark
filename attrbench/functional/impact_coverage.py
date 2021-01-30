from typing import Callable
from attrbench.lib.masking import ConstantMasker
import random
import torch
from os import path, listdir
from itertools import cycle
import re


def apply_patches(samples, labels, model, patch_folder):
    target_expr = re.compile(r".*_([0-9]*)\.pt")
    patch_names = cycle([filename for filename in listdir(patch_folder) if filename.endswith(".pt")])
    with torch.no_grad():
        original_output = model(samples).detach().cpu()
    successful = torch.zeros(samples.shape[0]).bool()
    attacked_samples = samples.clone()
    targets = torch.zeros(labels.shape).long()
    patch_mask = torch.zeros(samples.shape)
    max_tries = 50
    num_tries = 0
    while not torch.all(successful):
        # Load next patch
        num_tries += 1
        patch_name = next(patch_names)
        target = int(target_expr.match(patch_name).group(1))
        patch = torch.load(path.join(patch_folder, patch_name), map_location=lambda storage, loc: storage)
        image_size = samples.shape[-1]
        patch_size = patch.shape[-1]

        # Apply patch to all images in batch (random location, but same for each image in batch)
        indx = random.randint(0, image_size - patch_size)
        indy = random.randint(0, image_size - patch_size)
        attacked_samples[~successful, :, indx:indx + patch_size, indy:indy + patch_size] = patch.float()
        with torch.no_grad():
            adv_out = model(attacked_samples).detach().cpu()

        # Check which ones were successful now for the first time
        successful_now = ~successful & (original_output.argmax(axis=1) != target) & (adv_out.argmax(axis=1) == target) & (labels.cpu() != target)

        # Set the patch mask and targets for the samples that were successful this iteration
        patch_mask[successful_now, :, indx:indx + patch_size, indy:indy + patch_size] = 1
        targets[successful_now] = target

        # Add the currently successful samples to all successful samples
        successful = successful | successful_now

        if num_tries > max_tries:
            print(f"Not all samples could be attacked: {torch.sum(successful)}/{samples.size(0)} were successful.")
            break

    return attacked_samples, patch_mask, targets.to(samples.device)


def impact_coverage(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                    patch_folder=None, attacked_samples=None, patch_mask=None, targets=None,
                    debug_mode=False, writer=None):
    if len(samples.shape) != 4:
        raise ValueError("Impact Coverage can only be computed for image data and expects 4 input dimensions")

    # Argument validation: either provide patch folder, or provide attacked samples, patch mask, and targets.
    if patch_folder is None and (attacked_samples is None or patch_mask is None or targets is None):
        raise ValueError(f"If no patch folder is given, you must supply attacked_samples, patch_mask and targets.")
    if patch_folder is not None and not (attacked_samples is None and patch_mask is None and targets is None):
        print("patch_folder was provided, attacked_samples, patch_mask, targets will be ignored.")

    if patch_folder is not None:
        attacked_samples, patch_mask, targets = apply_patches(samples, labels, model, patch_folder)

    # Get attributions
    attrs = method(samples, target=targets).detach()
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
    nr_top_attributions = patch_mask.long().sum().item()

    # Create mask of critical factors (most important pixels/features according to attributions)
    to_mask = sorted_indices[:, -nr_top_attributions:]
    masker = ConstantMasker(feature_level="pixel" if attrs.shape[1] == 1 else "channel", mask_value=1.)
    critical_factor_mask = masker.mask(torch.zeros(attrs.shape), to_mask)

    # Calculate IoU of critical factors (top n attributions) with adversarial patch
    patch_mask_flattened = patch_mask.flatten(1).bool()
    critical_factor_mask_flattened = critical_factor_mask.flatten(1).bool()
    intersection = (patch_mask_flattened & critical_factor_mask_flattened).sum(dim=1)
    union = (patch_mask_flattened | critical_factor_mask_flattened).sum(dim=1)
    iou = intersection.float() / union.float()
    if debug_mode:
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)
    # [batch_size]
    return iou

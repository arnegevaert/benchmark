import random
import logging
import re
import os
from typing import Dict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from attribench._attribution_method import AttributionMethod
from attribench.result import ImpactCoverageResult
from attribench.result._grouped_batch_result import GroupedBatchResult
from attribench.data import IndexDataset
import torch
from itertools import cycle
import numpy as np


def _impact_coverage_batch(
    model: nn.Module,
    method_dict: Dict[str, AttributionMethod],
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    patch_folder: str,
    patch_names_cycle: cycle,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    target_expr = re.compile(r".*_([0-9]*)\.pt")
    batch_result: Dict[str, torch.Tensor] = {
        method_name: torch.zeros(1) for method_name in method_dict.keys()
    }
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # Get original output and initialize datastructures
    with torch.no_grad():
        original_output = model(batch_x).detach().cpu()
    successful = torch.zeros(batch_x.shape[0]).bool()
    attacked_samples = batch_x.clone()
    targets = torch.zeros(batch_y.shape).long()
    patch_mask = torch.zeros(batch_x.shape)
    max_tries = 50
    num_tries = 0

    # Apply patches to images
    while not torch.all(successful):
        num_tries += 1
        # Load next patch
        patch_name = next(patch_names_cycle)
        match_expr = target_expr.match(patch_name)
        if match_expr is None:
            raise ValueError(
                f"Patch name {patch_name} does not match" " expected format."
            )
        target = int(match_expr.group(1))
        patch = torch.load(
            os.path.join(patch_folder, patch_name),
            map_location=lambda storage, _: storage,
        ).to(device)
        image_size = batch_x.shape[-1]
        patch_size = patch.shape[-1]

        # Apply patch to all images in batch (random location,
        # but same for each image in batch)
        indx = random.randint(0, image_size - patch_size)
        indy = random.randint(0, image_size - patch_size)
        attacked_samples[~successful, ...] = batch_x[~successful, ...].clone()
        attacked_samples[
            ~successful,
            :,
            indx : indx + patch_size,
            indy : indy + patch_size,
        ] = patch.float()
        with torch.no_grad():
            adv_out = model(attacked_samples).detach().cpu()

        # Set the patch mask and targets for the samples that were
        # successful this iteration
        # We set the patch mask for all samples that weren't yet
        # successful
        # This way, if any samples can't be attacked,
        # they will still have a patch on them
        # (even though it didn't flip the prediction)
        patch_mask[~successful, ...] = 0
        patch_mask[
            ~successful,
            :,
            indx : indx + patch_size,
            indy : indy + patch_size,
        ] = 1
        targets[~successful] = target

        # Add the currently successful samples to all successful samples
        successful_now = (
            # Output was originally not equal to target
            (original_output.argmax(axis=1) != target)
            # Output is now equal to target
            & (adv_out.argmax(axis=1) == target)
            # Ground truth is not equal to target
            & (batch_y.cpu() != target)
        )
        successful = successful | successful_now

        if num_tries > max_tries:
            logging.warning(
                "Not all samples could be attacked:"
                f"{torch.sum(successful)}/{batch_x.size(0)}"
                " were successful."
            )
            break
    targets = targets.to(device)

    # Compute impact coverage for each method
    for method_name, method in method_dict.items():
        attrs = method(attacked_samples, targets).detach().cpu().numpy()

        # Check attributions shape
        if attrs.shape[1] not in (1, 3):
            raise ValueError(
                "Impact Coverage only works on image data."
                "Attributions must have 1 or 3 color channels."
                f"Found attributions shape {attrs.shape}."
            )
        # If attributions have only 1 color channel,
        # we need a single-channel patch mask as well
        if attrs.shape[1] == 1:
            patch_mask = patch_mask[:, 0, :, :]

        # Get indices of top k attributions
        flattened_attrs = attrs.reshape(attrs.shape[0], -1)
        sorted_indices = flattened_attrs.argsort()
        # Number of top attributions is equal to number of features
        # masked by the patch
        # We assume here that the mask is the same size for all samples!
        nr_top_attributions = patch_mask[0, ...].long().sum().item()

        # Create mask of critical factors (most important
        # pixels/features according to attributions)
        to_mask = sorted_indices[:, -nr_top_attributions:]
        critical_factor_mask = np.zeros(attrs.shape).reshape(
            attrs.shape[0], -1
        )
        batch_size = attrs.shape[0]
        batch_dim = np.tile(
            range(batch_size), (nr_top_attributions, 1)
        ).transpose()
        critical_factor_mask[batch_dim, to_mask] = 1
        critical_factor_mask = critical_factor_mask.astype(bool)

        # Calculate IoU of critical factors (top n attributions) with
        # adversarial patch
        patch_mask_flattened = patch_mask.flatten(1).bool().numpy()
        intersection = (patch_mask_flattened & critical_factor_mask).sum(
            axis=1
        )
        union = (patch_mask_flattened | critical_factor_mask).sum(axis=1)
        iou = intersection.astype(float) / union.astype(float)
        batch_result[method_name] = iou
    return batch_result


def impact_coverage(
    model: nn.Module,
    samples_dataset: Dataset,
    batch_size: int,
    method_dict: Dict[str, AttributionMethod],
    patch_folder: str,
    device: torch.device = torch.device("cpu"),
) -> ImpactCoverageResult:
    """Computes the Impact Coverage metric for a given dataset, model, and
    set of attribution methods.

    Impact Coverage is computed by applying an adversarial patch to the input.
    This patch causes the model to change its prediction.
    The Impact Coverage metric is the intersection over union (IoU) of the
    patch with the top n attributions of the input, where n is the number of
    features masked by the patch. The idea is that, as the patch causes the
    model to change its prediction, the corresponding region in the image
    should be highly relevant to the model's prediction.

    Impact Coverage requires a folder containing adversarial patches. The
    patches should be named as follows: patch_<target>.pt, where <target>
    is the target class of the patch. The target class is the class that
    the model will predict when the patch is applied to the input.

    To generate adversarial patches, the 
    :meth:`~attribench.functional.train_adversarial_patches` function
    or :class:`~attribench.distributed.TrainAdversarialPatches` class
    can be used.

    Parameters
    ----------
    model : nn.Module
        Model to compute Impact Coverage for.
    samples_dataset : Dataset
        Dataset to compute Impact Coverage for.
    batch_size : int
        Batch size to use when computing Impact Coverage.
    method_dict : Dict[str, AttributionMethod]
        Dictionary mapping method names to attribution methods.
    patch_folder : str
        Path to folder containing adversarial patches.
    device : torch.device, optional
        Device to use for computing Impact Coverage.
        Default: torch.device("cpu")

    Returns
    -------
    ImpactCoverageResult
        Result of the Impact Coverage metric.
    """
    # Get names of patches and compile regular expression for deriving
    # target labels
    patch_names = [
        filename
        for filename in os.listdir(patch_folder)
        if filename.endswith(".pt")
    ]
    patch_names_cycle = cycle(patch_names)

    index_dataset = IndexDataset(samples_dataset)
    dataloader = DataLoader(
        index_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    result = ImpactCoverageResult(list(method_dict.keys()), len(index_dataset))

    for batch_indices, batch_x, batch_y in dataloader:
        batch_result = _impact_coverage_batch(
            model,
            method_dict,
            batch_x,
            batch_y,
            patch_folder,
            patch_names_cycle,
            device,
        )
        result.add(GroupedBatchResult(batch_indices, batch_result))
    return result

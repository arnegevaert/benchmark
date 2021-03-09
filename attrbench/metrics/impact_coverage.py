from typing import Callable, Tuple, Dict, List

import h5py

from attrbench.lib.masking import ConstantMasker
from attrbench.lib import AttributionWriter
from attrbench.metrics import Metric, MetricResult
import random
import torch
from os import path, listdir
from itertools import cycle
import re
import numpy as np


def _apply_patches(samples: torch.Tensor, labels: torch.Tensor, model: Callable,
                   patch_folder: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        patch = torch.load(path.join(patch_folder, patch_name), map_location=lambda storage, loc: storage).to(samples.device)
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


def impact_coverage(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                    patch_folder: str, writer=None):
    if len(samples.shape) != 4:
        raise ValueError("Impact Coverage can only be computed for image data and expects 4 input dimensions")
    attacked_samples, patch_mask, targets = _apply_patches(samples, labels, model, patch_folder)
    return _compute_coverage(attacked_samples, method, patch_mask, targets, writer)


class ImpactCoverage(Metric):
    def __init__(self, model, methods: Dict[str, Callable], patch_folder: str, writer_dir: str = None):
        self.methods = methods
        super().__init__(model, list(methods.keys()), writer_dir)
        self.patch_folder = patch_folder
        self.writers = {method_name: path.join(writer_dir, method_name) if writer_dir else None
                        for method_name in methods}
        self.result = ImpactCoverageResult(list(methods.keys()))

    def run_batch(self, samples, labels, attrs_dict=None):
        attacked_samples, patch_mask, targets = _apply_patches(samples, labels,
                                                               self.model, self.patch_folder)
        for method_name in self.methods:
            method = self.methods[method_name]
            iou = _compute_coverage(attacked_samples, method, patch_mask,
                                    targets, writer=self._get_writer(method_name)).reshape(-1, 1)
            self.result.append(method_name, iou)


class ImpactCoverageResult(MetricResult):
    def __init__(self, method_names: List[str]):
        super().__init__(method_names)
        self.data = {m_name: [] for m_name in self.method_names}

    def add_to_hdf(self, group: h5py.Group):
        group.attrs["type"] = "ImpactCoverageResult"
        for method_name in self.method_names:
            group.create_dataset(method_name, data=torch.cat(self.data[method_name]).numpy())

    def append(self, method_name, batch):
        self.data[method_name].append(batch)

    @staticmethod
    def load_from_hdf(self, group: h5py.Group):
        method_names = list(group.keys())
        result = ImpactCoverageResult(method_names)
        result.data = {m_name: [group[m_name]] for m_name in method_names}

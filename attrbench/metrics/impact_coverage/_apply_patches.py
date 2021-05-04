import random
import re
from itertools import cycle
from os import path, listdir
from typing import Callable, Tuple
import logging

import torch


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
        patch = torch.load(path.join(patch_folder, patch_name), map_location=lambda storage, loc: storage).to(
            samples.device)
        image_size = samples.shape[-1]
        patch_size = patch.shape[-1]

        # Apply patch to all images in batch (random location, but same for each image in batch)
        indx = random.randint(0, image_size - patch_size)
        indy = random.randint(0, image_size - patch_size)
        attacked_samples[~successful, ...] = samples[~successful, ...].clone()
        attacked_samples[~successful, :, indx:indx + patch_size, indy:indy + patch_size] = patch.float()
        with torch.no_grad():
            adv_out = model(attacked_samples).detach().cpu()

        # Set the patch mask and targets for the samples that were successful this iteration
        # We set the patch mask for all samples that weren't yet successful
        # This way, if any samples can't be attacked, they will still have a patch on them
        # (even though it didn't flip the prediction)
        patch_mask[~successful, ...] = 0
        patch_mask[~successful, :, indx:indx + patch_size, indy:indy + patch_size] = 1
        targets[~successful] = target

        # Add the currently successful samples to all successful samples
        successful_now = (original_output.argmax(axis=1) != target) &\
                         (adv_out.argmax(axis=1) == target) &\
                         (labels.cpu() != target)
        successful = successful | successful_now

        if num_tries > max_tries:
            logging.info(
                f"Not all samples could be attacked: {torch.sum(successful)}/{samples.size(0)} were successful.")
            break
    return attacked_samples, patch_mask, targets.to(samples.device)

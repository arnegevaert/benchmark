from typing import Callable, List
from attrbench.lib import MaskingPolicy
import torch


# TODO do this more intelligently (using rough linear search + individual binary search)
# We assume none of the samples has the same label as the output of the network when given
# a fully masked image (in which case we might not see a flip)
def deletion_until_flip(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                        masking_policy: MaskingPolicy, step_size: float, debug_mode=False):
    if step_size < 0 or step_size > 0.5:
        raise ValueError("Step size must be between 0 and 0.5 (percentage of pixels)")
    debug_data = {}
    attrs = method(samples, labels).detach()
    if debug_mode:
        debug_data["attrs"] = attrs
        debug_data["orig_samples"] = samples
        debug_data["flipped_samples"] = [None for _ in range(samples.shape[0])]
    num_inputs = torch.prod(torch.tensor(attrs.shape[1:])).item()
    attrs = attrs.flatten(1)
    sorted_indices = attrs.argsort().cpu().detach().numpy()
    abs_step_size = max(1, int(step_size * num_inputs))

    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()
    mask_size = 0
    orig_predictions = model(samples)
    orig_predictions = torch.argmax(orig_predictions, dim=1)
    while not torch.all(flipped) and mask_size < num_inputs:
        mask_size += abs_step_size
        # Mask/insert pixels
        if mask_size == 0:
            masked_samples = samples
        else:
            masked_samples = masking_policy(samples, sorted_indices[:, -mask_size:])

        with torch.no_grad():
            predictions = torch.argmax(model(masked_samples), dim=1)
        criterion = (predictions != orig_predictions)
        new_flipped = torch.logical_or(flipped, criterion.cpu())
        flipped_this_iteration = (new_flipped != flipped)
        if debug_mode:
            for i in range(samples.shape[0]):
                if flipped_this_iteration[i]:
                    debug_data["flipped_samples"] = masked_samples[i]
        result[flipped_this_iteration] = mask_size
        flipped = new_flipped
    # Set maximum value for samples that were never flipped
    result[result == -1] = num_inputs
    if debug_mode:
        return result, debug_data
    return result
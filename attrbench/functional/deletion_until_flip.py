from typing import Callable
from attrbench.lib import MaskingPolicy
import torch


# We assume none of the samples has the same label as the output of the network when given
# a fully masked image (in which case we might not see a flip)
def deletion_until_flip(samples: torch.Tensor, model: Callable, attrs: torch.Tensor,
                        num_steps: float, masking_policy: MaskingPolicy, debug_mode=False, writer=None):
    debug_data = {}
    if debug_mode:
        debug_data["flipped_samples"] = [None for _ in range(samples.shape[0])]
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)
    num_inputs = torch.prod(torch.tensor(attrs.shape[1:])).item()
    attrs = attrs.flatten(1)
    sorted_indices = attrs.argsort().cpu().detach().numpy()
    total_features = attrs.shape[1]
    step_size = int(total_features / num_steps)
    if num_steps > total_features or num_steps < 2:
        raise ValueError(f"Number of steps must be between 2 and {total_features} (got {num_steps})")

    result = torch.tensor([-1 for _ in range(samples.shape[0])]).int()
    flipped = torch.tensor([False for _ in range(samples.shape[0])]).bool()
    mask_size = 0
    orig_predictions = model(samples)
    orig_predictions = torch.argmax(orig_predictions, dim=1)
    while not torch.all(flipped) and mask_size < num_inputs:
        mask_size += step_size
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
                    debug_data["flipped_samples"][i] = masked_samples[i]
        result[flipped_this_iteration] = mask_size
        flipped = new_flipped
    # Set maximum value for samples that were never flipped
    result[result == -1] = num_inputs
    if debug_mode:
        writer.add_images('Flipped samples', torch.stack(debug_data["flipped_samples"]))
    return result
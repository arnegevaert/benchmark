from typing import Callable, List
from attrbench.lib import MaskingPolicy
import torch


def insertion_deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                       mask_range: List[int], masking_policy: MaskingPolicy, mode: str, debug_mode=False):
    if mode not in ["deletion", "insertion"]:
        raise ValueError("Mode must be either deletion or insertion")
    debug_data = {}
    result = []
    attrs = method(samples, labels)  # [batch_size, *sample_shape]
    if debug_mode:
        debug_data["attrs"] = attrs
        debug_data["masked_samples"] = []
    # Flatten each sample in order to sort indices per sample
    attrs = attrs.flatten(1)  # [batch_size, -1]
    # Sort indices of attrs in ascending order
    sorted_indices = attrs.argsort().cpu().detach().numpy()

    for i in mask_range:
        # Mask/insert pixels
        if i == 0:
            if mode == "deletion":
                masked_samples = samples
            else:
                masked_samples = masking_policy(samples, sorted_indices)  # If i == 0, we insert no pixels, ie mask all pixels
        else:
            to_mask = sorted_indices[:, -i:] if mode == "deletion" else sorted_indices[:, :-i]
            masked_samples = masking_policy(samples, to_mask)

        if debug_mode:
            debug_data["masked_samples"].append(masked_samples)
        # Get predictions for result
        with torch.no_grad():
            predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
        result.append(predictions)
    result = torch.cat(result, dim=1).cpu()  # [batch_size, len(mask_range)]
    if debug_mode:
        return result, debug_data
    return result
from typing import Callable, List
from attrbench.lib.util import mask_pixels, insert_pixels
import torch


_MASK_METHODS = {
    "deletion": mask_pixels,
    "insertion": insert_pixels
}


def insertion_deletion_curves(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                              mask_range: List[int], mask_value: float, mode: str):
    assert mode in ["deletion", "insertion"]
    device = samples.device

    result = []
    attrs = method(samples, labels)  # [batch_size, *sample_shape]
    pixel_level = attrs.size(1) == 1
    # Flatten each sample in order to sort indices per sample
    attrs = attrs.flatten(1)  # [batch_size, -1]
    # Sort indices of attrs in ascending order
    sorted_indices = attrs.argsort().cpu().detach().numpy()

    for i in mask_range:
        # Mask/insert pixels
        if i > 0:
            masked_samples = _MASK_METHODS[mode](samples, sorted_indices[:, -i:], mask_value, pixel_level)
        else:
            masked_samples = samples if mode == "deletion" else torch.ones(samples.shape).to(device) * mask_value
        # Get predictions for result
        with torch.no_grad():
            predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
        result.append(predictions)
    return torch.cat(result, dim=1).cpu()  # [batch_size, len(mask_range)]

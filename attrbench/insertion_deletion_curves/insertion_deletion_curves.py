from typing import Callable, List
from util import mask_pixels, insert_pixels
import torch


_MASK_METHODS = {
    "deletion": mask_pixels,
    "insertion": insert_pixels
}


def insertion_deletion_curves(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                              mask_range: List[int], mask_value: float, pixel_level_mask: bool,
                              device: str, mode: str):
    assert mode in ["deletion", "insertion"]
    samples = samples.to(device)
    labels = labels.to(device)
    # Check which samples are classified correctly, only want to calculate for correctly classified samples
    with torch.no_grad():
        y_pred = torch.argmax(model(samples), dim=1)
    samples = samples[y_pred == labels]
    labels = labels[y_pred == labels]
    batch_size = samples.size(0)
    if batch_size > 0:
        result = []
        attrs = method(samples, labels)  # [batch_size, *sample_shape]
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.flatten(1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        sorted_indices = attrs.argsort().cpu().detach().numpy()

        for i in mask_range:
            # Mask/insert pixels
            if i > 0:
                masked_samples = _MASK_METHODS[mode](samples, sorted_indices[:, -i:], mask_value, pixel_level_mask)
            else:
                masked_samples = samples if mode == "deletion" else torch.ones(samples.shape).to(device) * mask_value
            # Get predictions for result
            with torch.no_grad():
                predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
            result.append(predictions.cpu().detach())
        # [batch_size, len(mask_range)]
        return torch.cat(result, dim=1)

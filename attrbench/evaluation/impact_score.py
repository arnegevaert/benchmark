from typing import List, Callable
from lib.util import mask_pixels
import torch


def impact_score(samples: torch.Tensor, labels: torch.Tensor, model: Callable, mask_range: List[int], method: Callable,
                 mask_value: float, strict: bool, tau: float = None, device: str = "cpu"):
    if not (strict or tau):
        raise ValueError("Provide value for tau when calculating non-strict impact score")
    counts = []

    samples = samples.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.no_grad():
        orig_out = model(samples).gather(dim=1, index=labels.view(-1, 1))
    correctly_classified = torch.argmax(orig_out, dim=1) == labels
    batch_size = correctly_classified.sum().item()
    samples = samples[correctly_classified]
    orig_out = orig_out[correctly_classified]
    labels = labels[correctly_classified]

    if batch_size > 0:
        attrs = method(samples, target=labels)
        pixel_level_attrs = len(attrs.shape) == 3
        attrs = attrs.flatten(1)
        sorted_indices = attrs.argsort().cpu()
        for n_idx, n in enumerate(mask_range):
            masked_samples = mask_pixels(samples, sorted_indices[:, -n:], mask_value, pixel_level_attrs)
            with torch.no_grad():
                masked_out = model(masked_samples)
            confidence = masked_out.gather(dim=1, index=labels.view(-1, 1))
            flipped = torch.argmax(masked_out, dim=1) != labels
            if not strict:
                flipped = flipped | confidence <= orig_out * tau
            counts.append(flipped.sum().item())
        return torch.tensor(counts) / batch_size
    return None

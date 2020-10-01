from typing import List, Callable
from attrbench.lib.util import mask_pixels
import torch
from torch.nn.functional import softmax


def impact_score(samples: torch.Tensor, labels: torch.Tensor, model: Callable, mask_range: List[int], method: Callable,
                 mask_value: float, strict: bool, tau: float = None):
    if not (strict or tau):
        raise ValueError("Provide value for tau when calculating non-strict impact score")
    counts = []

    with torch.no_grad():
        orig_out = model(samples)
    correctly_classified = torch.argmax(orig_out, dim=1) == labels
    batch_size = correctly_classified.sum().item()
    samples = samples[correctly_classified]
    orig_out = orig_out[correctly_classified]
    labels = labels[correctly_classified]

    orig_confidence = softmax(orig_out).gather(dim=1, index=labels.view(-1, 1))
    if batch_size > 0:
        attrs = method(samples, target=labels)
        pixel_level_attrs = len(attrs.shape) == 3
        attrs = attrs.flatten(1)
        sorted_indices = attrs.argsort().cpu()
        for n_idx, n in enumerate(mask_range):
            masked_samples = mask_pixels(samples, sorted_indices[:, -n:], mask_value, pixel_level_attrs)
            with torch.no_grad():
                masked_out = model(masked_samples)
            confidence = softmax(masked_out).gather(dim=1, index=labels.view(-1, 1))
            flipped = torch.argmax(masked_out, dim=1) != labels
            if not strict:
                flipped = flipped | (confidence <= orig_confidence * tau).squeeze()
            counts.append(flipped.sum().item())
        # [len(mask_range)], int
        return torch.tensor(counts), batch_size
    return 0, 0

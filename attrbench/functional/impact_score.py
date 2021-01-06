from attrbench.lib import MaskingPolicy
from typing import List, Callable
import torch
from torch.nn.functional import softmax


def impact_score(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable, mask_range: List[int],
                 strict: bool, masking_policy: MaskingPolicy, tau: float = None, debug_mode=False):
    if not (strict or tau):
        raise ValueError("Provide value for tau when calculating non-strict impact score")
    counts = []
    debug_data = {}

    with torch.no_grad():
        orig_out = model(samples)
    correctly_classified = torch.argmax(orig_out, dim=1) == labels
    batch_size = correctly_classified.sum().item()
    samples = samples[correctly_classified]
    orig_out = orig_out[correctly_classified]
    labels = labels[correctly_classified]

    orig_confidence = softmax(orig_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
    if batch_size > 0:
        attrs = method(samples, target=labels).detach()
        if debug_mode:
            debug_data["attrs"] = attrs
            debug_data["masked_samples"] = []
        attrs = attrs.flatten(1)
        sorted_indices = attrs.argsort().cpu()
        for n in mask_range:
            masked_samples = masking_policy(samples, sorted_indices[:, -n:]) if n > 0 else samples.clone()
            if debug_mode:
                debug_data["masked_samples"].append(masked_samples)
            with torch.no_grad():
                masked_out = model(masked_samples)
            confidence = softmax(masked_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
            flipped = torch.argmax(masked_out, dim=1) != labels
            if not strict:
                flipped = flipped | (confidence <= orig_confidence * tau).squeeze()
            counts.append(flipped.sum().item())
        # [len(mask_range)]
        result = torch.tensor(counts)
        if debug_mode:
            return result, batch_size, debug_data
        return result, batch_size
    if debug_mode:
        return torch.tensor([]), 0, {}
    return torch.tensor([]), 0

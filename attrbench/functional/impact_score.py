from attrbench.lib.masking import Masker
from typing import Callable
import torch
import numpy as np
from torch.nn.functional import softmax


def impact_score(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.tensor, num_steps: int,
                 strict: bool, masker: Masker, tau: float = None, debug_mode=False, writer=None):
    if not (strict or tau):
        raise ValueError("Provide value for tau when calculating non-strict impact score")
    counts = []
    masker.initialize_baselines(samples)

    with torch.no_grad():
        orig_out = model(samples)
    correctly_classified = torch.argmax(orig_out, dim=1) == labels
    batch_size = correctly_classified.sum().item()
    samples = samples[correctly_classified]
    orig_out = orig_out[correctly_classified]
    labels = labels[correctly_classified]

    orig_confidence = softmax(orig_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
    if batch_size > 0:
        assert attrs.shape[0] == samples.shape[0]
        if debug_mode:
            writer.add_images('Image samples', samples)
            writer.add_images('attributions', attrs)

        attrs = attrs.flatten(1)
        sorted_indices = attrs.argsort().cpu()
        total_features = attrs.shape[1]
        mask_range = list((np.linspace(0, 1, num_steps) * total_features).astype(np.int))
        for n in mask_range:
            if n > 0:
                masked_out, masked_samples = masker.predict_masked(samples, sorted_indices[:, -n:],
                                                                   model, return_masked_samples=True)
            else:
                masked_samples = samples.clone()
                masked_out = orig_out
            if debug_mode:
                writer.add_images('Masked samples', masked_samples, global_step=n)
            confidence = softmax(masked_out, dim=1).gather(dim=1, index=labels.view(-1, 1))
            flipped = torch.argmax(masked_out, dim=1) != labels
            if not strict:
                flipped = flipped | (confidence <= orig_confidence * tau).squeeze()
            counts.append(flipped.sum().item())
        # [len(mask_range)]
        result = torch.tensor(counts)
        return result, batch_size
    if debug_mode:
        return torch.tensor([]), 0, {}
    return torch.tensor([]), 0

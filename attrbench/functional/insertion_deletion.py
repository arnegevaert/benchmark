from typing import Callable, List
from attrbench.lib import MaskingPolicy
import torch
import numpy as np


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
              num_steps: int, masking_policy: MaskingPolicy, debug_mode=False, writer=None):
    return _insertion_deletion(samples, labels, model, method, num_steps, masking_policy, "insertion",
                               debug_mode=debug_mode, writer=writer)


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
             num_steps: int, masking_policy: MaskingPolicy, debug_mode=False, writer=None):
    return _insertion_deletion(samples, labels, model, method, num_steps, masking_policy, "deletion",
                               debug_mode=debug_mode, writer=writer)


def _insertion_deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                        num_steps: int, masking_policy: MaskingPolicy, mode: str, debug_mode: bool=False,
                        writer=None):
    if mode not in ["deletion", "insertion"]:
        raise ValueError("Mode must be either deletion or insertion")
    result = []
    attrs = method(samples, labels).detach()  # [batch_size, *sample_shape]
    if debug_mode:
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)

    # Flatten each sample in order to sort indices per sample
    attrs = attrs.flatten(1)  # [batch_size, -1]
    # Sort indices of attrs in ascending order
    sorted_indices = attrs.argsort().cpu().detach().numpy()

    total_features = attrs.shape[1]
    mask_range = list((np.linspace(0, 1, num_steps) * total_features).astype(np.int))
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
            writer.add_images('masked samples', masked_samples, global_step=i)
        # Get predictions for result
        with torch.no_grad():
            predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
        result.append(predictions)
    result = torch.cat(result, dim=1).cpu()  # [batch_size, len(mask_range)]

    return result
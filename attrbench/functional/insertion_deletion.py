from typing import Callable, List
from attrbench.lib import MaskingPolicy
import torch


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
              mask_range: List[int], masking_policy: MaskingPolicy,attrs, debug_mode=False, writer=None):
    return _insertion_deletion(samples, labels, model, method, mask_range, masking_policy, "insertion", attrs,
                               debug_mode=debug_mode, writer = writer)


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
             mask_range: List[int], masking_policy: MaskingPolicy,attrs, debug_mode=False, writer=None):
    return _insertion_deletion(samples, labels, model, method, mask_range, masking_policy, "deletion", attrs,
                               debug_mode=debug_mode, writer = writer)


def _insertion_deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, method: Callable,
                        mask_range: List[int], masking_policy: MaskingPolicy, mode: str,attrs, debug_mode: bool=False,
                        writer=None):
    if mode not in ["deletion", "insertion"]:
        raise ValueError("Mode must be either deletion or insertion")
    debug_data = {}
    result = []
    if attrs is None:
        attrs = method(samples, labels).detach()  # [batch_size, *sample_shape]
    if debug_mode:
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)

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
            writer.add_images('masked samples', masked_samples, global_step=i)
        # Get predictions for result
        with torch.no_grad():
            predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
        result.append(predictions)
    result = torch.cat(result, dim=1).cpu()  # [batch_size, len(mask_range)]

    return result
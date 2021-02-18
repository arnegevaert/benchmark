from typing import Callable
from attrbench.lib.masking import Masker
import torch
import numpy as np


def insertion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
              num_steps: int, masker: Masker, writer=None):
    sorted_indices, orig_predictions, neutral_predictions, mask_range = _init(samples, labels, model, attrs,
                                                                              num_steps, masker, writer)
    result = []
    for i in mask_range:
        if i == 0:
            predictions = neutral_predictions
            masked_samples = masker.mask(samples, sorted_indices)
        else:
            predictions, masked_samples = masker.predict_masked(samples, sorted_indices[:, :-i],
                                                                model, return_masked_samples=True)
            predictions = predictions.gather(dim=1, index=labels.unsqueeze(-1))
        if writer is not None:
            writer.add_images('masked samples', masked_samples, global_step=i)
        result.append((predictions - neutral_predictions) / orig_predictions)
    result = torch.cat(result, dim=1).cpu()  # [batch_size, len(mask_range)]
    return result


def deletion(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
             num_steps: int, masker: Masker, writer=None):
    sorted_indices, orig_predictions, _, mask_range = _init(samples, labels, model, attrs,
                                                            num_steps, masker, writer)
    result = []
    for i in mask_range:
        if i == 0:
            predictions = orig_predictions
            masked_samples = samples
        else:
            predictions, masked_samples = masker.predict_masked(samples, sorted_indices[:, -i:],
                                                                model, return_masked_samples=True)
            predictions = predictions.gather(dim=1, index=labels.unsqueeze(-1))
        if writer is not None:
            writer.add_images('masked samples', masked_samples, global_step=i)
        result.append((predictions - orig_predictions) / orig_predictions)
    result = torch.cat(result, dim=1).cpu()  # [batch_size, len(mask_range)]
    return result


def _init(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
          num_steps: int, masker: Masker, writer=None):
    if writer is not None:
        writer.add_images('Image samples', samples)
        writer.add_images('attributions', attrs)
    masker.initialize_baselines(samples)

    # Flatten each sample in order to sort indices per sample
    attrs = attrs.flatten(1)  # [batch_size, -1]
    # Sort indices of attrs in ascending order
    sorted_indices = attrs.argsort().cpu().detach().numpy()

    # Get original predictions
    with torch.no_grad():
        orig_predictions = model(samples).gather(dim=1, index=labels.unsqueeze(-1))
        neutral_predictions = masker.predict_masked(samples, sorted_indices, model) \
            .gather(dim=1, index=labels.unsqueeze(-1))

    total_features = attrs.shape[1]
    mask_range = list((np.linspace(0, 1, num_steps) * total_features).astype(np.int))
    return sorted_indices, orig_predictions, neutral_predictions, mask_range

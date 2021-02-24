from typing import Callable
from attrbench.lib.masking import Masker
import torch
import numpy as np
from attrbench.lib import mask_segments, segment_samples_attributions


def _init(samples, attrs, writer, masker):
    # Segment images and attributions
    segmented_images, avg_attrs = segment_samples_attributions(samples.detach().cpu().numpy(),
                                                               attrs.detach().cpu().numpy())
    if writer is not None:
        writer.add_images("segmented samples", segmented_images)

    # Initialize masker
    masker.initialize_baselines(samples)

    # Sort segment attribution values
    sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]
    return segmented_images, avg_attrs, sorted_indices


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
         masker: Masker, writer=None):
    segmented_images, avg_attrs, sorted_indices = _init(samples, attrs, writer, masker)

    # Get original predictions
    with torch.no_grad():
        orig_predictions = model(samples).gather(dim=1, index=labels.unsqueeze(-1))

    # Iteratively mask the k most important segments
    preds = []
    for i in range(sorted_indices.shape[1]+1):
        if i == 0:
            masked_samples = samples
            predictions = orig_predictions
        else:
            masked_samples = mask_segments(samples, segmented_images, sorted_indices[:, -i:], masker)
            with torch.no_grad():
                predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
        if writer is not None:
            writer.add_images('masked samples', masked_samples, global_step=i)
        preds.append(predictions / orig_predictions)
    preds = torch.cat(preds, dim=1).cpu()

    # Calculate AOC for each sample (depends on how many segments each sample had)
    aoc = []
    for i in range(samples.shape[0]):
        num_segments = len(np.unique(segmented_images[i, ...]))
        aoc.append(1 - np.trapz(preds[i, :num_segments+1], x=np.linspace(0, 1, num_segments+1)))

    return torch.tensor(aoc).unsqueeze(-1)  # [batch_size, 1]


def iiof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
         masker: Masker, writer=None):
    segmented_images, avg_attrs, sorted_indices = _init(samples, attrs, writer, masker)

    # Get original and neutral predictions
    with torch.no_grad():
        orig_predictions = model(samples).gather(dim=1, index=labels.unsqueeze(-1))
        fully_masked = mask_segments(samples, segmented_images, sorted_indices, masker)
        neutral_predictions = model(fully_masked).gather(dim=1, index=labels.unsqueeze(-1))

    # Iteratively mask the k most important segments
    preds = []
    for i in range(sorted_indices.shape[1]+1):
        if i == 0:
            masked_samples = fully_masked
            predictions = neutral_predictions
        else:
            masked_samples = mask_segments(samples, segmented_images, sorted_indices[:, :-i], masker)
            with torch.no_grad():
                predictions = model(masked_samples).gather(dim=1, index=labels.unsqueeze(-1))
        if writer is not None:
            writer.add_images('masked samples', masked_samples, global_step=i)
        preds.append(predictions / orig_predictions)
    preds = torch.cat(preds, dim=1).cpu()

    # Calculate AUC for each sample (depends on how many segments each sample had)
    auc = []
    for i in range(samples.shape[0]):
        num_segments = len(np.unique(segmented_images[i, ...]))
        auc.append(np.trapz(preds[i, :num_segments+1], x=np.linspace(0, 1, num_segments+1)))

    return torch.tensor(auc).unsqueeze(-1)  # [batch_size, 1]

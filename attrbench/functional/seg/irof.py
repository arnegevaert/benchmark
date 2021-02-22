from typing import Callable
from attrbench.lib.masking import Masker
import torch
import numpy as np
from attrbench.functional.seg.util import mask_segments, segment_samples_attributions


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
         masker: Masker, writer=None):
    # Segment images and attributions
    segmented_images, avg_attrs = segment_samples_attributions(samples.detach().cpu().numpy(),
                                                               attrs.detach().cpu().numpy())

    # Initialize masker
    masker.initialize_baselines(samples)

    # Sort segment attribution values
    sorted_indices = avg_attrs.argsort()  # [batch_size, num_segments]

    # Get original and neutral predictions
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

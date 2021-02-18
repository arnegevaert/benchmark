from typing import Callable
from attrbench.lib.masking import Masker
from skimage.segmentation import slic
import torch
import numpy as np


def _mask_segments(images: np.ndarray, seg_images: np.ndarray, segments: np.ndarray, masker: Masker) -> np.ndarray:
    if not (images.shape[0] == seg_images.shape[0] and images.shape[0] == segments.shape[0] and
            images.shape[-2:] == seg_images.shape[-2:]):
        raise ValueError(f"Incompatible shapes: {images.shape}, {seg_images.shape}, {segments.shape}")
    bool_masks = []
    for i in range(images.shape[0]):
        seg_img = seg_images[i, ...]
        segs = segments[i, ...]
        bool_masks.append(np.isin(seg_img, segs))
    bool_masks = torch.tensor(np.stack(bool_masks, axis=0)).long()
    return masker.mask_boolean(images, bool_masks)


def _calculage_segment_attributions(seg_images: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    segments = np.unique(seg_images)
    seg_img_flat = seg_images.reshape(seg_images.shape[0], -1)
    attrs_flat = attrs.reshape(attrs.shape[0], -1)
    avg_attrs = np.zeros((seg_images.shape[0], len(segments)))
    for i, seg in enumerate(segments):  # Segments should be 0, ..., n, but we use enumerate just in case
        mask = (seg_img_flat == seg).astype(np.long)
        masked_attrs = mask * attrs_flat
        mean_attrs = np.sum(masked_attrs, axis=1) / np.sum(mask, axis=1)
        # If seg does not exist for image, mean_attrs will be nan. Replace with -inf.
        avg_attrs[:, i] = np.nan_to_num(mean_attrs, nan=-np.inf)
    return avg_attrs


def irof(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
         masker: Masker, writer=None):
    # Segment images using SLIC
    segmented_images = np.stack([slic(np.transpose(samples[i, ...].detach().cpu().numpy(), (1, 2, 0)), start_label=0)
                                 for i in range(samples.shape[0])])
    segmented_images = np.expand_dims(segmented_images, axis=1)
    segments = np.unique(segmented_images)

    # Initialize masker
    masker.initialize_baselines(samples)

    # Calculate average attribution for each segment in each image
    avg_attrs = _calculage_segment_attributions(segmented_images, attrs.cpu().detach().numpy())

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
            masked_samples = _mask_segments(samples, segmented_images, sorted_indices[:, -i:], masker)
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

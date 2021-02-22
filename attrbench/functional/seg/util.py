import numpy as np
import torch
from attrbench.lib.masking import Masker
from skimage.segmentation import slic
from typing import Tuple


def mask_segments(images: np.ndarray, seg_images: np.ndarray, segments: np.ndarray, masker: Masker) -> np.ndarray:
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


def segment_samples_attributions(samples: np.ndarray, attrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Segment images using SLIC
    seg_images = np.stack([slic(np.transpose(samples[i, ...], (1, 2, 0)),
                                start_label=0, slic_zero=True)
                           for i in range(samples.shape[0])])
    seg_images = np.expand_dims(seg_images, axis=1)

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
    return seg_images, avg_attrs

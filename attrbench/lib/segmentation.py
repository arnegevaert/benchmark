import numpy as np
import torch
from attrbench.lib.masking import Masker
from skimage.segmentation import slic
from typing import Tuple, List


def isin(a: torch.tensor, b: torch.tensor):
    # https://stackoverflow.com/questions/60918304/get-indices-of-elements-in-tensor-a-that-are-present-in-tensor-b
    return (a[..., None] == b).any(-1)


def mask_segments(images: torch.tensor, seg_images: torch.tensor, segments: List[torch.tensor], masker: Masker) -> torch.tensor:
    if not (images.shape[0] == seg_images.shape[0] and images.shape[0] == len(segments) and
            images.shape[-2:] == seg_images.shape[-2:]):
        raise ValueError(f"Incompatible shapes: {images.shape}, {seg_images.shape}, {len(segments)}")
    bool_masks = []
    for i in range(images.shape[0]):
        seg_img = seg_images[i, ...]
        segs = segments[i]
        bool_masks.append(isin(seg_img, segs))
    bool_masks = torch.stack(bool_masks, dim=0)
    return masker.mask_boolean(images, bool_masks)


def segment_samples(samples: np.ndarray) -> np.ndarray:
    # Segment images using SLIC
    seg_images = np.stack([slic(np.transpose(samples[i, ...], (1, 2, 0)),
                                start_label=0, slic_zero=True)
                           for i in range(samples.shape[0])])
    seg_images = np.expand_dims(seg_images, axis=1)
    return seg_images


def segment_attributions(seg_images: torch.tensor, attrs: torch.tensor) -> torch.tensor:
    segments = torch.unique(seg_images)
    seg_img_flat = seg_images.flatten(1)
    attrs_flat = attrs.flatten(1)
    avg_attrs = torch.zeros((seg_images.shape[0], segments.max() + 1), device=seg_images.device)
    for seg in segments:
        mask = (seg_img_flat == seg).long()
        masked_attrs = mask * attrs_flat
        mask_size = torch.sum(mask, dim=1)
        sum_attrs = torch.sum(masked_attrs, dim=1)
        mean_attrs = torch.true_divide(sum_attrs, mask_size)
        # If seg does not exist for image, mean_attrs will be inf or nan (since mask_size=0). Replace with -inf.
        mean_attrs[~torch.isfinite(mean_attrs)] = -np.inf
        avg_attrs[:, seg] = mean_attrs
    return avg_attrs

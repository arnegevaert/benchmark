import numpy as np
import torch
from attrbench.lib.masking import Masker
from skimage.segmentation import slic
from typing import Tuple, List


def isin(a: np.ndarray, b: np.ndarray):
    # https://stackoverflow.com/questions/60918304/get-indices-of-elements-in-tensor-a-that-are-present-in-tensor-b
    return (a[..., None] == b).any(-1)


def mask_segments(images: torch.tensor, seg_images: np.ndarray, segments: List[List[int]], masker: Masker) -> torch.tensor:
    if not (images.shape[0] == seg_images.shape[0] and images.shape[0] == len(segments) and
            images.shape[-2:] == seg_images.shape[-2:]):
        raise ValueError(f"Incompatible shapes: {images.shape}, {seg_images.shape}, {len(segments)}")
    bool_masks = []
    for i in range(images.shape[0]):
        seg_img = seg_images[i, ...]
        segs = np.array(segments[i])
        bool_masks.append(isin(seg_img, segs))
    bool_masks = np.stack(bool_masks, axis=0)
    return masker.mask_boolean(images, torch.tensor(bool_masks))


def segment_samples(samples: np.ndarray) -> np.ndarray:
    # Segment images using SLIC
    seg_images = np.stack([slic(np.transpose(samples[i, ...], (1, 2, 0)),
                                start_label=0, slic_zero=True)
                           for i in range(samples.shape[0])])
    seg_images = np.expand_dims(seg_images, axis=1)
    return seg_images


def segment_attributions(seg_images: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    segments = np.unique(seg_images)
    seg_img_flat = seg_images.reshape(seg_images.shape[0], -1)
    attrs_flat = attrs.reshape(attrs.shape[0], -1)
    avg_attrs = np.zeros((seg_images.shape[0], np.max(segments) + 1))
    for seg in segments:
        mask = (seg_img_flat == seg).astype(int)
        masked_attrs = mask * attrs_flat
        mask_size = np.sum(mask, axis=1)
        sum_attrs = np.sum(masked_attrs, axis=1)
        mean_attrs = np.true_divide(sum_attrs, mask_size)
        # If seg does not exist for image, mean_attrs will be inf (since mask_size=0). Replace with -inf.
        mean_attrs[mean_attrs == np.inf] = -np.inf
        avg_attrs[:, seg] = mean_attrs
    return avg_attrs


def segment_samples_attributions(samples: np.ndarray, attrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seg_images = segment_samples(samples)
    avg_attrs = segment_attributions(seg_images, attrs)
    return seg_images, avg_attrs

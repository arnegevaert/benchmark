import numpy as np
from attrbench.lib.masking import Masker
from skimage.segmentation import slic
from typing import Tuple


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
    avg_attrs = np.zeros((seg_images.shape[0], len(segments)))
    for i, seg in enumerate(segments):  # Segments should be 0, ..., n, but we use enumerate just in case
        mask = (seg_img_flat == seg).astype(np.long)
        masked_attrs = mask * attrs_flat
        mask_size = np.sum(mask, axis=1)
        sum_attrs = np.sum(masked_attrs, axis=1)
        mean_attrs = np.divide(sum_attrs, mask_size, out=np.zeros_like(sum_attrs), where=mask_size!=0)
        # If seg does not exist for image, mean_attrs will be nan. Replace with -inf.
        avg_attrs[:, i] = np.nan_to_num(mean_attrs, nan=-np.inf)
    return avg_attrs


def segment_samples_attributions(samples: np.ndarray, attrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seg_images = segment_samples(samples)
    avg_attrs = segment_attributions(seg_images, attrs)
    return seg_images, avg_attrs

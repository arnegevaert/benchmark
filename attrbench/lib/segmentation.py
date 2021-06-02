import numpy as np
import torch
from skimage.segmentation import slic


def isin(a: torch.tensor, b: torch.tensor):
    # https://stackoverflow.com/questions/60918304/get-indices-of-elements-in-tensor-a-that-are-present-in-tensor-b
    return (a[..., None] == b).any(-1)


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
    result = np.zeros((seg_images.shape[0], len(segments)))
    for seg in segments:
        mask = (seg_img_flat == seg).astype(np.long)
        masked_attrs = mask * attrs_flat
        mask_size = np.sum(mask, axis=1)
        sum_attrs = np.sum(masked_attrs, axis=1)
        mean_attrs = np.divide(sum_attrs, mask_size, out=np.zeros_like(sum_attrs), where=mask_size != 0)
        # If seg does not exist for image, mask_size == 0. Set to -inf.
        mean_attrs[mask_size == 0] = -np.inf
        result[:, seg] = mean_attrs
    return result

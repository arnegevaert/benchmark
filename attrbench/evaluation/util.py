import torch.nn.functional as F
import numpy as np
import torch


transform_fns = {
    "identity": lambda l: l,
    "softmax": lambda l: F.softmax(l, dim=1),
}


def mask_pixels(imgs, indices, mask_value, pixel_level_mask):
    batch_size, color_channels = imgs.shape[:2]
    num_pixels = indices.shape[1]
    result = imgs.clone().to(imgs.device, non_blocking=True)
    batch_dim = np.column_stack([range(batch_size) for _ in range(num_pixels)])
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        for color_dim in range(color_channels):
            result[(batch_dim, color_dim, *unraveled)] = mask_value
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = mask_value
    return result


def insert_pixels(imgs, indices, mask_value, pixel_level_mask):
    num_pixels = indices.shape[1]
    batch_size, color_channels = imgs.shape[:2]
    result = torch.ones(imgs.shape).to(imgs.device) * mask_value
    batch_dim = np.column_stack([range(batch_size) for _ in range(num_pixels)])
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        for color_dim in range(color_channels):
            result[(batch_dim, color_dim, *unraveled)] = imgs[(batch_dim, color_dim, *unraveled)]
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = imgs[(batch_dim, *unraveled)]
    return result

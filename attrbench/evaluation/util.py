import torch
import numpy as np


def logit_softmax(logits):
    batch_size, n_classes = logits.shape
    sub_logits = logits - torch.max(logits, dim=1)[0].unsqueeze(-1)  # For numerical stability
    repeat_sub_logits = sub_logits.repeat(1, n_classes).view(batch_size, n_classes, n_classes)
    selector = torch.diag(torch.ones(n_classes)).bool()\
        .repeat(batch_size, 1)\
        .view(batch_size, n_classes, n_classes)

    lse = torch.log(torch.sum(torch.exp(repeat_sub_logits[~selector].view(batch_size, n_classes, n_classes-1)), dim=2))
    return sub_logits - lse


def mask_top_pixels(imgs, indices, mask_value, pixel_level_mask):
    batch_size, color_channels = imgs.shape[:2]
    num_pixels = indices.shape[1]
    result = imgs.clone()
    batch_dim = np.column_stack([range(batch_size) for _ in range(num_pixels)])
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        for color_dim in range(color_channels):
            result[(batch_dim, color_dim, *unraveled)] = mask_value
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = mask_value
    return result


def insert_top_pixels(imgs, indices, mask_value, pixel_level_mask):
    num_pixels = indices.shape[1]
    batch_size, color_channels = imgs.shape[:2]
    result = torch.ones(imgs.shape) * mask_value
    batch_dim = np.column_stack([range(batch_size) for _ in range(num_pixels)])
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        for color_dim in range(color_channels):
            result[(batch_dim, color_dim, *unraveled)] = imgs[(batch_dim, color_dim, *unraveled)]
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = imgs[(batch_dim, *unraveled)]
    return result

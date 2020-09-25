import numpy as np
import torch


def sum_of_attributions(attrs, indices):
    attrs = attrs.flatten(1)
    mask_attrs = attrs.gather(dim=1, index=indices)
    return mask_attrs.sum(dim=1, keepdim=True)


# TODO these functions still form a GPU bottleneck
def mask_pixels(imgs, indices, mask_value, pixel_level_mask):
    num_pixels = indices.shape[1]
    result = imgs.clone().to(imgs.device, non_blocking=True)
    batch_dim = np.tile(range(imgs.shape[0]), (num_pixels, 1)).transpose()
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        result[batch_dim, :, unraveled[0], unraveled[1]] = mask_value
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = mask_value
    return result


def insert_pixels(imgs, indices, mask_value, pixel_level_mask):
    num_pixels = indices.shape[1]
    result = torch.ones(imgs.shape).to(imgs.device, non_blocking=True) * mask_value
    batch_dim = np.tile(range(imgs.shape[0]), (num_pixels, 1)).transpose()
    if pixel_level_mask:
        unraveled = np.unravel_index(indices, imgs.shape[2:])
        result[batch_dim, :, unraveled[0], unraveled[1]] = imgs[batch_dim, :, unraveled[0], unraveled[1]]
    else:
        unraveled = np.unravel_index(indices, imgs.shape[1:])
        result[(batch_dim, *unraveled)] = imgs[(batch_dim, *unraveled)]
    return result


if __name__ == "__main__":
    # Test mask/insert pixels
    imgs = torch.randn((4, 3, 5, 5))
    i = torch.tensor([
        [17, 3, 2],
        [20, 5, 12],
        [1, 3, 2],
        [0, 6, 9]
    ])
    masked = mask_pixels(imgs, i, 5, True)

    # Test sum of attributions
    attrs = torch.arange(100).reshape(4, 5, 5)
    i = torch.tensor([
        [17, 3, 2],
        [20, 5, 12],
        [1, 3, 2],
        [0, 6, 9]
    ])
    s = sum_of_attributions(attrs, i)
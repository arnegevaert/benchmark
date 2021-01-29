import numpy as np
import torch


def get_index(shape, index):
    res = []
    for i in range(len(shape)):
        res.append(int(index // np.prod(shape[i+1:])))
        index = index % np.prod(shape[i+1:])
    return tuple(res)


def generate_images_indices(shape, mask_size, feature_level):
    assert feature_level in ("channel", "pixel")
    images = torch.randn(shape)
    max_idx = shape[-1] * shape[-2]
    if feature_level == "channel":
        max_idx *= shape[-3]
    indices = torch.stack([
        torch.tensor(
            np.random.choice(max_idx, size=mask_size, replace=False)
        ) for _ in range(shape[0])
    ], dim=0)
    return images, indices
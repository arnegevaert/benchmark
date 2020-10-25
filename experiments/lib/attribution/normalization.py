import torch
from typing import Callable

class Normalization:
    def __init__(self, base_method: Callable):
        self.base_method = base_method

    def __call__(self, x, target):
        attrs = self.base_method(x, target)
        abs_attrs = torch.abs(attrs.flatten(1))
        max_abs_attr_per_image = torch.max(abs_attrs, dim=1)[0]
        if torch.any(max_abs_attr_per_image == 0):
            print("Warning: completely 0 attributions returned for sample.")
            # If an image has 0 max abs attr, all attrs are 0 for that image
            # Divide by 1 to return the original constant 0 attributions
            max_abs_attr_per_image[torch.where(max_abs_attr_per_image == 0)] = 1.0
        # Add as many singleton dimensions to max_abs_attr_per_image as necessary to divide
        while len(max_abs_attr_per_image.shape) < len(attrs.shape):
            max_abs_attr_per_image = torch.unsqueeze(max_abs_attr_per_image, dim=-1)
        normalized = attrs / max_abs_attr_per_image
        return normalized.reshape(attrs.shape)

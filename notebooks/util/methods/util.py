import numpy as np
import warnings


def normalize_attributions(attrs):
    abs_attrs = np.abs(attrs.reshape(attrs.shape[0], -1))
    max_abs_attr_per_image = np.max(abs_attrs, axis=1)
    if np.any(max_abs_attr_per_image == 0):
        warnings.warn("Completely 0 attributions returned for sample.")
        # If an image has 0 max abs attr, all attrs are 0 for that image
        # Divide by 1 to return the original constant 0 attributions
        max_abs_attr_per_image[np.where(max_abs_attr_per_image == 0)] = 1.0
    # Add as many singleton dimensions to max_abs_attr_per_image as necessary to divide
    while len(max_abs_attr_per_image.shape) < len(attrs.shape):
        max_abs_attr_per_image = np.expand_dims(max_abs_attr_per_image, axis=-1)
    normalized = attrs / max_abs_attr_per_image
    return normalized.reshape(attrs.shape)

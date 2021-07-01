from captum import attr
import captum
import torch
import numpy as np
from skimage import segmentation
import math


class ImageLime:
    def __init__(self, model):
        self.model = model
        self.method = None

    def __call__(self, x, target):
        # If this is the first call, the method hasn't been set yet.
        # This is because the kernel width depends on the number of dimensions.
        if self.method is None:
            num_features = x.flatten(1).shape[1]
            kernel_width = 0.75 * math.sqrt(num_features)
            sim_fn = attr._core.lime.get_exp_kernel_similarity_function(
                distance_mode="euclidean", kernel_width=kernel_width)
            # Need to pass lasso manually, current captum implementation uses alpha=1. This should be alpha=0.01
            lasso = captum._utils.models.linear_model.SkLearnLasso(alpha=0.01)
            self.method = attr.Lime(self.model, similarity_func=sim_fn, interpretable_model=lasso)

        # Segment the images using SLIC
        images = x.detach().cpu().numpy()
        num_channels = images.shape[1]
        masks = []
        for i in range(images.shape[0]):
            img = np.transpose(images[i], (1,2,0))
            mask = segmentation.slic(img)
            masks.append(mask)
        masks = torch.tensor(data=masks, device=x.device, dtype=torch.long)
        masks = masks.unsqueeze(dim=1).expand(-1, num_channels, -1, -1)

        # Next, compute LIME. Default value for n_samples of 1000 comes from official LIME implementation
        return self.method.attribute(x, target=target, feature_mask=masks, n_samples=25)

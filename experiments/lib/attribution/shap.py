from captum import attr
from skimage import segmentation
import numpy as np
import torch

class Shap:
    def __init__(self, model, n_segments):
        self.n_segments=n_segments
        self.method = attr.ShapleyValues(model)
    def __call__(self, x, target):
        masks = get_super_pixels(x,self.n_segments)
        return self.method.attribute(x, target=target, feature_mask= masks)


def get_super_pixels(x,k):
    images = x.detach().cpu().numpy()
    nr_of_channels = images.shape[1] # assuming grayscale images have 1 channel
    masks = []
    for i in range(images.shape[0]):
        input_image = np.transpose(images[i],(1, 2, 0))
        mask=segmentation.slic(input_image,n_segments=k,slic_zero=True)
        masks.append(mask)
    masks = torch.LongTensor(np.stack(masks))
    masks = masks.unsqueeze(dim=1)
    return masks.expand(-1,nr_of_channels,-1,-1).to(x.device)

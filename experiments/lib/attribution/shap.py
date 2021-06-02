from captum import attr
from captum._utils.models.linear_model import SkLearnLinearModel
from skimage import segmentation
import numpy as np
import torch
from torch.utils.data import DataLoader

class TabularShap:
    def __init__(self, model, n_samples, baselines=None,feature_mask=None):
        self.method = attr.ShapleyValueSampling(model)
        self.n_samples = n_samples
        if isinstance(baselines, list):
            baselines = torch.Tensor(baselines)[None]
        self.baselines=baselines
        self.feature_mask=feature_mask

    def __call__(self,x, target):
        return self.method.attribute(x,baselines=self.baselines,target=target,feature_mask=self.feature_mask)

class TabularLime:
    def __init__(self, model, n_samples, baselines=None, feature_mask=None):
        self.method = attr.Lime(model,interpretable_model=SkLearnLinearModel("linear_model.LinearRegression")) # defaolt lasso model always learns to output 0.
        self.n_samples = n_samples
        if isinstance(baselines, list):
            baselines = torch.Tensor(baselines)[None]
        self.baselines=baselines
        self.feature_mask=feature_mask

    def __call__(self,x, target):
        return self.method.attribute(x,baselines=self.baselines,target=target,feature_mask=self.feature_mask)

class KernelShap:
    def __init__(self, model, n_samples, super_pixels=True, n_segments=None, feature_mask=None, baselines=None):
        if super_pixels and n_samples is None:
            raise ValueError(f"n_segments cannot be None when using super_pixels")
        self.n_segments = n_segments
        self.super_pixels = super_pixels
        self.method = attr.KernelShap(model)
        self.n_samples = n_samples
        self.feature_mask = feature_mask
        if isinstance(baselines, list):
            baselines = torch.Tensor(baselines)[None]
        self.baselines=baselines

    def __call__(self, x, target):
        if self.feature_mask is not None:
            masks=self.feature_mask
        else:
            masks = get_super_pixels(x, self.n_segments) if self.super_pixels else None
        return self.method.attribute(x, target=target, feature_mask=masks, n_samples=self.n_samples, baselines=self.baselines)


class DeepShap:
    def __init__(self, model, reference_dataset, n_baseline_samples):
        self.method = attr.DeepLift(model)
        self.n_baseline_samples = n_baseline_samples
        self.reference_dataset = reference_dataset
        self.ref_sampler = DataLoader(
            dataset=self.reference_dataset,
            batch_size=self.n_baseline_samples,
            shuffle=True, drop_last=True)

    def _get_reference_batch(self):
        return next(iter(self.ref_sampler))[0]

    def __call__(self, x, target):
        baseline = self._get_reference_batch().to(x.device)

        dl_attr = (x * 0).cpu()
        for base in baseline:
            dl_attr += self.method.attribute(x, target=target, baselines=base[None]).detach().cpu()
        attr = dl_attr / self.n_baseline_samples

        return attr


def get_super_pixels(x, k):
    images = x.detach().cpu().numpy()
    nr_of_channels = images.shape[1]  # assuming grayscale images have 1 channel
    masks = []
    for i in range(images.shape[0]):
        input_image = np.transpose(images[i], (1, 2, 0))
        mask = segmentation.slic(input_image, n_segments=k, slic_zero=True, start_label=0)
        masks.append(mask)
    masks = torch.LongTensor(np.stack(masks))
    masks = masks.unsqueeze(dim=1)
    return masks.expand(-1, nr_of_channels, -1, -1).to(x.device)

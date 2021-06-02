from attrbench.lib.masking import ImageMasker
import torch


class ConstantMasker(ImageMasker):
    def __init__(self, feature_level, mask_value=0.):
        super().__init__(feature_level)
        self.mask_value = mask_value

    def initialize_baselines(self, samples: torch.tensor):
        self.baseline = torch.ones(samples.shape, device=samples.device) * self.mask_value

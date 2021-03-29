from attrbench.lib.masking import ImageMasker
import numpy as np
import torch



class ConstantMasker(ImageMasker):
    def __init__(self, samples, attributions, feature_level, mask_value=0., segmented_samples: np.ndarray =None):
        super().__init__(samples, attributions, feature_level, segmented_samples)
        self.mask_value = mask_value
        self.initialize_baselines(samples)


    def initialize_baselines(self, samples: torch.tensor):
        self.baseline = torch.ones(samples.shape, device=samples.device) * self.mask_value

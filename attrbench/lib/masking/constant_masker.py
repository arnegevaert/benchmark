from attrbench.lib.masking import Masker
import numpy as np


class ConstantMasker(Masker):
    def __init__(self,samples, attributions, feature_level, mask_value=0.,segmentation: np.ndarray =None):
        super().__init__(samples, attributions,feature_level,segmentation)
        self.mask_value = mask_value
        self.initialize_baselines(samples)


    def initialize_baselines(self, samples):
        self.baseline = np.ones(samples.shape) * self.mask_value

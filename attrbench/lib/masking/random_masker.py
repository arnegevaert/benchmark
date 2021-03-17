from attrbench.lib.masking import Masker
import torch
import numpy as np


class RandomMasker(Masker):
    def __init__(self,samples, attributions, feature_level, std=1, num_samples=1,segmentation: np.ndarray =None):
        super().__init__(samples, attributions,feature_level,segmentation)
        self.std = std
        self.num_samples = num_samples
        self.initialize_baselines(samples)

    def initialize_baselines(self, samples):
        self.baseline = np.random.normal(loc=0, scale=self.std, size=samples.shape)

from attrbench.lib.masking import Masker
import torch


class SampleAverageMasker(Masker):
    def initialize_baselines(self, samples):
        batch_size, num_channels, rows, cols = samples.shape
        self.baseline = torch.mean(samples.flatten(2), dim=-1)\
            .reshape(batch_size, num_channels, 1, 1)\
            .repeat(1, 1, rows, cols)

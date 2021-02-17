from attrbench.lib.masking import Masker
from cv2 import blur
import torch


class BlurringMasker(Masker):
    def __init__(self, feature_level, kernel_size):
        super().__init__(feature_level)
        if not 0 < kernel_size < 1.0:
            raise ValueError("Kernel size is expressed as a fraction of image height, and must be between 0 and 1.")
        self.kernel_size = kernel_size

    def initialize_baselines(self, samples):
        kernel_size = int(self.kernel_size * samples.shape[-1])

        baseline = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...]
            cv_sample = sample.permute(1, 2, 0).cpu().numpy()
            blurred_sample = torch.tensor(blur(cv_sample, (kernel_size, kernel_size)))
            if len(blurred_sample.shape) == 2:
                blurred_sample = blurred_sample.unsqueeze(-1)
            baseline.append(blurred_sample.permute(2, 0, 1).to(samples.device))
        self.baseline = torch.stack(baseline, dim=0)


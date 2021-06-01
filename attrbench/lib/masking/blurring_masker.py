from attrbench.lib.masking import ImageMasker
from cv2 import blur
import numpy as np
import torch


class BlurringMasker(ImageMasker):
    def __init__(self, feature_level, kernel_size):
        super().__init__(feature_level)
        if not 0 < kernel_size < 1.0:
            raise ValueError("Kernel size is expressed as a fraction of image height, and must be between 0 and 1.")
        self.kernel_size = kernel_size

    def initialize_baselines(self, samples: torch.tensor):
        kernel_size = int(self.kernel_size * samples.shape[-1])

        baseline = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...].cpu().numpy()
            cv_sample = np.transpose(sample, (1, 2, 0))
            blurred_sample = blur(cv_sample, (kernel_size, kernel_size))
            if len(blurred_sample.shape) == 2:
                blurred_sample = blurred_sample[..., np.newaxis]
            baseline.append(np.transpose(blurred_sample, (2, 0, 1)))
        self.baseline = torch.tensor(np.stack(baseline, axis=0), device=samples.device)

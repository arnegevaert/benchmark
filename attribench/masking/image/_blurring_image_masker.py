from attribench.masking.image import ImageMasker
from cv2 import blur
import numpy as np
import torch


class BlurringImageMasker(ImageMasker):
    """Image masker that masks pixels or features by replacing them with
    a blurred version of the image. The amount of blurring is controlled
    by the ``kernel_size`` parameter, which is expressed as a fraction
    of the image height.
    """
    def __init__(self, masking_level: str, kernel_size: float):
        """
        Parameters
        ----------
        feature_level : str
            The level at which to mask the image. Must be either
            ``"pixel"`` or ``"feature"``.
        kernel_size : float
            Kernel size for the blurring operation, expressed as a fraction
            of the image height. Must be between 0 and 1.

        Raises
        ------
        ValueError
            If ``kernel_size`` is not between 0 and 1.
        """
        super().__init__(masking_level)
        if not 0 < kernel_size < 1.0:
            raise ValueError(
                "Kernel size is expressed as a fraction of image height,"
                " and must be between 0 and 1."
            )
        self.kernel_size = kernel_size

    def _initialize_baselines(self, samples: torch.Tensor):
        kernel_size = int(self.kernel_size * samples.shape[-1])

        baseline = []
        for i in range(samples.shape[0]):
            sample = samples[i, ...].cpu().numpy()
            cv_sample = np.transpose(sample, (1, 2, 0))
            blurred_sample = blur(cv_sample, (kernel_size, kernel_size))
            if len(blurred_sample.shape) == 2:
                blurred_sample = blurred_sample[..., np.newaxis]
            baseline.append(np.transpose(blurred_sample, (2, 0, 1)))
        self.baseline = torch.tensor(
            np.stack(baseline, axis=0), device=samples.device
        )

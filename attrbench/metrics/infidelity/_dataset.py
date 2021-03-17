import numpy as np
from skimage.segmentation import slic
import torch


class _Perturbation:
    def __init__(self, samples: torch.tensor, perturbation_size, num_perturbations):
        self.samples = samples
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations
        self.rng = np.random.default_rng()

    def __call__(self):
        raise NotImplementedError


class _GaussianPerturbation(_Perturbation):
    # perturbation_size is stdev of noise
    def __call__(self):
        return torch.randn(*self.samples.shape, device=self.samples.device) * self.perturbation_size


class _SquareRemovalPerturbation(_Perturbation):
    # perturbation_size is (square height)/(image height)
    def __call__(self):
        height = self.samples.shape[2]
        width = self.samples.shape[3]
        square_size_int = int(self.perturbation_size * height)
        x_loc = self.rng.integers(0, width - square_size_int, size=1).item()
        y_loc = self.rng.integers(0, height - square_size_int, size=1).item()
        perturbation_mask = torch.zeros(self.samples.shape, device=self.samples.device)
        perturbation_mask[:, :, x_loc:x_loc + square_size_int, y_loc:y_loc + square_size_int] = 1
        perturbation_vector = self.samples * perturbation_mask
        return perturbation_vector


class _SegmentRemovalPerturbation(_Perturbation):
    # perturbation size is number of segments
    def __init__(self, samples, perturbation_size, num_perturbations):
        super().__init__(samples, perturbation_size, num_perturbations)
        samples = samples.cpu().numpy()
        seg_samples = np.stack([slic(np.transpose(samples[i, ...], (1, 2, 0)),
                                     start_label=0, slic_zero=True)
                                for i in range(samples.shape[0])])
        self.seg_samples = np.expand_dims(seg_samples, axis=1)

    def __call__(self):
        perturbed_samples, perturbation_vectors = [], []
        # This needs to happen per sample, since samples don't necessarily have
        # the same number of segments
        # TODO check if this can happen better on GPU (look at segsensn)
        for i in range(self.samples.shape[0]):
            seg_sample = self.seg_samples[i, ...]
            sample = self.samples[i, ...]
            # Get all segment numbers
            all_segments = np.unique(seg_sample)
            # Select segments to mask
            segments_to_mask = self.rng.choice(all_segments, self.perturbation_size, replace=False)
            # Create boolean mask of pixels that need to be removed
            to_remove = torch.tensor(np.isin(seg_sample, segments_to_mask), device=self.samples.device).float()
            # Create perturbation vector by multiplying mask with image
            perturbation_vector = sample * to_remove
            perturbation_vectors.append(perturbation_vector)
        return torch.stack(perturbation_vectors, dim=0)

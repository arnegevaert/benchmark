import numpy as np
from attrbench.lib import segment_samples, isin
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
        segmented_images = segment_samples(samples.cpu().numpy())
        self.segmented_images = torch.tensor(segmented_images, device=samples.device)
        self.segments = [np.unique(segmented_images[i, ...]) for i in range(samples.shape[0])]
        self.rng = np.random.default_rng()

    def __call__(self):
        perturbation_vectors = []
        # Select segments to mask for each sample
        segments_to_mask = torch.tensor(
            np.stack([self.rng.choice(self.segments[i], size=self.perturbation_size, replace=False)
                      for i in range(self.samples.shape[0])]), device=self.samples.device)
        for i in range(self.samples.shape[0]):
            seg_sample = self.segmented_images[i, ...]
            sample = self.samples[i, ...]
            # Create boolean mask of pixels that need to be removed
            to_remove = isin(seg_sample, segments_to_mask[i, ...])
            # Create perturbation vector by multiplying mask with image
            perturbation_vector = sample * to_remove
            perturbation_vectors.append(perturbation_vector)
        return torch.stack(perturbation_vectors, dim=0)

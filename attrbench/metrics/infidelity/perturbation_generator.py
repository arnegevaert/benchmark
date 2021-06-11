import numpy as np
from attrbench.lib import segment_samples, isin
import torch


class PerturbationGenerator:
    def __init__(self):
        self.samples = None
        self.rng = np.random.default_rng()

    def set_samples(self, samples):
        self.samples = samples

    def __call__(self):
        if self.samples is None:
            raise ValueError("Base samples must be set to generate perturbations")
        return self._generate_perturbation_vectors()

    def _generate_perturbation_vectors(self):
        raise NotImplementedError


class NoisyBaselinePerturbationGenerator(PerturbationGenerator):
    def __init__(self, sd):
        super().__init__()
        self.sd = sd

    # perturbation_size is stdev of noise
    def _generate_perturbation_vectors(self):
        # I = x - (x_0 + \epsilon) where x_0 = 0
        return self.samples - torch.randn(*self.samples.shape, device=self.samples.device) * self.sd


class GaussianPerturbationGenerator(PerturbationGenerator):
    def __init__(self, sd):
        super().__init__()
        self.sd = sd

    # perturbation_size is stdev of noise
    def _generate_perturbation_vectors(self):
        # I \sim \mathcal{N}(0, self.sd)
        return torch.randn(*self.samples.shape, device=self.samples.device) * self.sd


class SquarePerturbationGenerator(PerturbationGenerator):
    def __init__(self, square_size):
        super().__init__()
        self.square_size = square_size

    # perturbation_size is (square height)/(image height)
    def _generate_perturbation_vectors(self):
        height = self.samples.shape[2]
        width = self.samples.shape[3]
        x_loc = self.rng.integers(0, width - self.square_size, size=1).item()
        y_loc = self.rng.integers(0, height - self.square_size, size=1).item()
        perturbation_mask = torch.zeros(self.samples.shape, device=self.samples.device)
        perturbation_mask[:, :, x_loc:x_loc + self.square_size, y_loc:y_loc + self.square_size] = 1
        perturbation_vector = self.samples * perturbation_mask
        return perturbation_vector


class SegmentRemovalPerturbationGenerator(PerturbationGenerator):
    # perturbation size is number of segments
    def __init__(self, num_segments):
        super().__init__()
        self.num_segments = num_segments

    def set_samples(self, samples: torch.tensor):
        self.samples = samples
        segmented_images = segment_samples(samples.cpu().numpy())
        self.segmented_images = torch.tensor(segmented_images, device=samples.device)
        self.segments = [np.unique(segmented_images[i, ...]) for i in range(samples.shape[0])]
        self.rng = np.random.default_rng()

    def _generate_perturbation_vectors(self):
        perturbation_vectors = []
        # Select segments to mask for each sample
        segments_to_mask = torch.tensor(
            np.stack([self.rng.choice(self.segments[i], size=self.num_segments, replace=False)
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

import numpy as np
import torch

from attrbench.lib import AttributionWriter
from attrbench.lib import mask_segments, segment_samples, segment_attributions
from attrbench.lib.masking import Masker


class _DeletionDataset:
    def __init__(self, mode: str, start: float, stop: float, num_steps: int, samples: torch.tensor, attrs: np.ndarray,
                 masker: Masker):
        if mode not in ("morf", "lerf"):
            raise ValueError("Mode must be morf or lerf")
        if not ((0. <= start <= 1.) and (0. <= stop <= 1.)):
            raise ValueError("Start and stop must be between 0 and 1")
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        self.sorted_indices = np.argsort(attrs)
        if mode == "lerf":
            self.sorted_indices = self.sorted_indices[:, ::-1]

        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(start, stop, num_steps) * total_features).astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        if num_to_mask == 0:
            return self.samples.detach().clone()
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples


class _IrofDataset:
    def __init__(self, mode: str, start: float, stop: float, num_steps: int, samples: torch.tensor, masker: Masker,
                 writer: AttributionWriter = None):
        self.mode = mode
        self.start = start
        self.stop = stop
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        self.sorted_indices = None
        self.mask_ranges = None
        # Override sorted_indices to use segment indices instead of pixel indices
        self.segmented_images = segment_samples(samples.cpu().numpy())
        if writer is not None:
            writer.add_images("segmented samples", self.segmented_images)

    def set_attrs(self, attrs: np.ndarray):
        avg_attrs = segment_attributions(self.segmented_images, attrs)
        sorted_indices = []
        mask_ranges = []
        for i in range(self.segmented_images.shape[0]):
            # For each image, sort the indices separately
            cur_sorted_indices = avg_attrs[i, ...].argsort()
            # Count how many times -inf was present (these are non-existing segments)
            num_infs = np.count_nonzero(avg_attrs[i, ...] == -np.inf)
            # Remove the indices of non-existing segments, these are the first num_infs entries in the sorted indices
            cur_sorted_indices = cur_sorted_indices[num_infs:]
            sorted_indices.append(cur_sorted_indices if self.mode == "morf" else cur_sorted_indices[::-1])
            # Compute corresponding mask range for this image
            num_segments = len(cur_sorted_indices)
            mask_ranges[i] = list((np.linspace(self.start, self.stop, self.num_steps) * num_segments).astype(np.int))
        self.sorted_indices = sorted_indices  # List[np.ndarray] (varying number of segments for each image)
        self.mask_ranges = mask_ranges

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        indices = []
        for i in range(self.samples.shape[0]):
            # Get number of segments to mask for this image
            num_to_mask = self.mask_ranges[i][item]
            # Get num_segments most important (if morf) or least important (if lerf) segments
            indices.append(self.sorted_indices[i][-num_to_mask:])
        return mask_segments(self.samples, self.segmented_images, indices, self.masker)

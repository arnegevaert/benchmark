import numpy as np
import torch


class _MinimalSubsetDataset:
    def __init__(self, num_steps, samples: torch.tensor, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        self.masker.initialize_baselines(samples)
        # Flatten each sample in order to sort indices per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        # Sort indices of attrs in ascending order
        self.sorted_indices = np.argsort(attrs)
        total_features = attrs.shape[1]
        self.step_size = int(total_features / num_steps)
        if num_steps > total_features or num_steps < 2:
            raise ValueError(f"Number of steps must be between 2 and {total_features} (got {num_steps})")

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        num_to_mask = self.step_size * (item + 1)
        if num_to_mask > self.sorted_indices.shape[1]:
            raise StopIteration()
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)
        return masked_samples, num_to_mask


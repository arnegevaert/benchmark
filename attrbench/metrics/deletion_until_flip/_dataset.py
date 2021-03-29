import numpy as np
import torch


class _DeletionUntilFlipDataset:
    def __init__(self, num_steps, samples: torch.tensor, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        self.total_features = self.masker.get_total_features() # get_total_features = sorted_indices.shape[1]
        self.step_size = int(self.total_features / num_steps)
        if num_steps > self.total_features or num_steps < 2:
            raise ValueError(f"Number of steps must be between 2 and {self.total_features} (got {num_steps})")

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        num_to_mask = self.step_size * (item + 1)
        if num_to_mask > self.total_features:
            raise StopIteration()
        masked_samples= self.masker.mask_top(num_to_mask)
        return masked_samples, num_to_mask

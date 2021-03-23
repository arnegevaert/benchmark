from torch.utils.data import Dataset
import numpy as np


class _DeletionUntilFlipDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        total_features = self.masker.get_total_features()
        self.step_size = int(total_features / num_steps)
        if num_steps > total_features or num_steps < 2:
            raise ValueError(f"Number of steps must be between 2 and {total_features} (got {num_steps})")

    def __len__(self):
        return self.num_steps

    def __getitem__(self, item):
        num_to_mask = self.step_size * (item + 1)
        masked_samples= self.masker.mask_top(num_to_mask)
        return masked_samples, num_to_mask

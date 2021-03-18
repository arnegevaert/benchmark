from torch.utils.data import Dataset
import numpy as np


class _DeletionUntilFlipDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
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
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)

        masked_samples2 = self.masker.mask_top(num_to_mask)
        assert ((masked_samples == masked_samples2).all())
        return masked_samples, num_to_mask


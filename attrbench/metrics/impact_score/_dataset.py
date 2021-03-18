import numpy as np
from torch.utils.data import Dataset


class _ImpactScoreDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        attrs = attrs.reshape(attrs.shape[0], -1)
        self.sorted_indices = np.argsort(attrs)
        total_features = attrs.shape[1]
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        indices = self.sorted_indices[:, -num_to_mask:]
        masked_samples = self.masker.mask(self.samples, indices)

        masked_samples2 = self.masker.mask_top(num_to_mask)
        assert ((masked_samples == masked_samples2).all())
        return masked_samples


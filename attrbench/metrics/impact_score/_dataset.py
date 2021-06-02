import numpy as np
from torch.utils.data import Dataset


class _ImpactScoreDataset(Dataset):
    def __init__(self, num_steps, samples: np.ndarray, attrs: np.ndarray, masker):
        self.num_steps = num_steps
        self.samples = samples
        masker_constructor, masker_kwargs=masker
        self.masker = masker_constructor(samples, attrs,**masker_kwargs)
        total_features = self.masker.get_num_features()
        self.mask_range = list((np.linspace(0, 1, num_steps) * total_features)[1:].astype(np.int))

    def __len__(self):
        return len(self.mask_range)

    def __getitem__(self, item):
        num_to_mask = self.mask_range[item]
        masked_samples = self.masker.mask_top(num_to_mask)
        return masked_samples


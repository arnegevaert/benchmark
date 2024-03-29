import torch
from attribench.masking import Masker
from abc import abstractmethod


class MinimalSubsetDataset:
    def __init__(
        self,
        num_steps,
        samples: torch.Tensor,
        attrs: torch.Tensor,
        masker: Masker,
    ):
        self.num_steps = num_steps
        self.samples = samples
        self.masker = masker
        masker.set_batch(samples, attrs)
        self.total_features = self.masker.get_num_features()
        self.step_size = int(self.total_features / num_steps)
        if num_steps > self.total_features or num_steps < 2:
            raise ValueError(
                f"Number of steps must be between 2 and {self.total_features}"
                f" (got {num_steps})"
            )

    def __len__(self):
        return self.num_steps

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError


class MinimalSubsetDeletionDataset(MinimalSubsetDataset):
    def __getitem__(self, item):
        # Mask the k most important pixels
        num_to_mask = self.step_size * item
        if num_to_mask > self.total_features:
            raise StopIteration
        masked_samples = self.masker.mask_top(num_to_mask)
        return masked_samples, num_to_mask


class MinimalSubsetInsertionDataset(MinimalSubsetDataset):
    def __getitem__(self, item):
        # Mask the n-k least important pixels
        num_to_insert = self.step_size * item
        num_to_mask = self.total_features - num_to_insert
        if num_to_mask > self.total_features:
            raise StopIteration
        masked_samples = self.masker.mask_bot(num_to_mask)
        return masked_samples, num_to_insert

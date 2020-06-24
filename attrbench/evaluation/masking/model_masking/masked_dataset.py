import torch
import numpy as np
import torch.nn as nn
from typing import Iterable
from .masked_neural_network import MaskedInputLayer


def median_of_minima(data: Iterable, mask_fn: nn.Module):
    mins = []
    for samples, labels in iter(data):
        samples = mask_fn(samples).reshape(samples.shape[0], -1).detach().numpy()
        mins.append(np.min(samples, axis=1))
    mins = np.concatenate(mins)
    return np.median(mins)


class MaskedDataset:
    class Loader:
        def __init__(self, base_loader, masking_layer, med_of_min):
            self.base_loader = base_loader
            self.masking_layer = masking_layer
            self.med_of_min = med_of_min

        def __iter__(self):
            for samples, _ in iter(self.base_loader):
                yield self._gen_labels(samples)

        def __len__(self):
            return len(self.base_loader)

        def _gen_labels(self, samples):
            masked_samples = self.masking_layer(samples)
            reshaped_samples = masked_samples.reshape(masked_samples.shape[0], -1)
            labels = torch.all(reshaped_samples > self.med_of_min, dim=1).long()
            return samples, labels

    def __init__(self, train_loader, test_loader,
                 radius: int, mask_value: float, med_of_min: float = None):
        self._train_loader = train_loader
        self._test_loader = test_loader
        sample_shape = next(iter(self._train_loader))[0].shape[1:]
        self.masking_layer = MaskedInputLayer(sample_shape, radius, mask_value)
        self.med_of_min = med_of_min
        if not med_of_min:
            self.med_of_min = median_of_minima(self._train_loader(), self.masking_layer)
        print(f"Median of minima is {self.med_of_min}")

    def get_dataloader(self, train=True):
        base = self._train_loader if train else self._test_loader
        return MaskedDataset.Loader(base, self.masking_layer, self.med_of_min)

    def get_mask(self):
        return self.masking_layer.mask

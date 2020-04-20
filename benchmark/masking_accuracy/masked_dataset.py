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
    def __init__(self, train_data: Iterable, test_data: Iterable,
                 radius: int, mask_value: float, med_of_min: float = None):
        self.train_data = train_data
        self.test_data = test_data
        sample_shape = next(iter(train_data))[0].shape[1:]
        self.masking_layer = MaskedInputLayer(sample_shape, radius, mask_value)
        self.med_of_min = med_of_min
        if not med_of_min:
            self.med_of_min = median_of_minima(train_data, self.masking_layer)
        print(f"Median of minima is {self.med_of_min}")

    def get_train_data(self):
        for samples, _ in iter(self.train_data):
            yield self._gen_labels(samples)

    def get_test_data(self):
        for samples, _ in iter(self.test_data):
            yield self._gen_labels(samples)

    def _gen_labels(self, samples):
        masked_samples = self.masking_layer(samples)
        reshaped_samples = masked_samples.reshape(masked_samples.shape[0], -1)
        labels = torch.all(reshaped_samples > self.med_of_min, dim=1).long()
        return samples, labels

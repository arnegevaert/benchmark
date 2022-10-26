from torch.utils.data import Dataset
from typing import Callable
import h5py


class AttributionsDataset(Dataset):
    """
    File
    - method_1: [num_samples, *sample_shape]
    - method_2: [num_samples, *sample_shape]
    - ...
    """

    def __init__(self, samples_dataset: Dataset, path: str, aggregate_fn: Callable = None):
        self.path = path
        self.samples_dataset: Dataset = samples_dataset
        self.attributions_file: h5py.File | None = None
        self.aggregate_fn = aggregate_fn
        with h5py.File(path, "r") as fp:
            self.num_samples = fp.attrs["num_samples"]
            self.attribution_methods = list(fp.keys())

    def __getitem__(self, index):
        if self.attributions_file is None:
            self.attributions_file = h5py.File(self.path, "r")
        method_name = self.attribution_methods[index // self.num_samples]
        sample_idx = index % self.num_samples
        sample, label = self.samples_dataset[sample_idx]
        attrs = self.attributions_file[method_name][sample_idx]
        if self.aggregate_fn is not None:
            attrs = self.aggregate_fn(attrs)
        return sample_idx, sample, label, attrs, method_name

    def __len__(self):
        if self.attributions_file is None:
            self.attributions_file = h5py.File(self.path, "r")
        return self.num_samples * len(self.attribution_methods)

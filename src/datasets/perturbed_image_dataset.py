import h5py
import torch
import json
import numpy as np
from os import path
from typing import Iterator

from models import Model


class PerturbedImageDataset:
    def __init__(self, data_location, name, batch_size):
        self.name = name
        self.data_location = data_location
        self.batch_size = batch_size
        self.filename = path.join(data_location, name, "dataset.hdf5")
        self.metadata_filename = path.join(data_location, name, "meta.json")
        self.file = h5py.File(self.filename, 'r')
        self.metadata = json.load(open(self.metadata_filename))

    def _iterator(self, l):
        dataset = self.file[f"level_{l}"]
        count = 0
        while count + self.batch_size <= dataset.shape[0]:
            print(count + self.batch_size)
            yield dataset[count:count+self.batch_size, :]
            count += self.batch_size

    def get_levels(self):
        return self.metadata["perturbation_levels"]

    def get_fn(self):
        return self.metadata["perturbation_fn"]

    def get_iterators(self):
        return [
            self._iterator(level) for level in range(len(self.get_levels()))
        ]

    @staticmethod
    def generate(data_location, name, data_iterator: Iterator, model: Model,
                 perturbation_fn="noise", perturbation_levels=np.linspace(0, 1, 10), max_tries=10):
        assert(perturbation_fn in ["noise", "mean_shift"])
        p_fns = {
            "noise": lambda s, l: s + (np.random.rand(*s.shape)) * l,
            "mean_shift": lambda s, l: s + l
        }
        filename = path.join(data_location, name, "dataset.hdf5")
        metadata_filename = path.join(data_location, name, "meta.json")
        file = h5py.File(filename, 'w')
        datasets_created = False
        row_count = 0
        batch_size = None
        # [batch_size, *sample_shape], [batch_size]
        for samples, labels in data_iterator:
            # Convert to numpy if necessary
            if type(samples) == torch.Tensor:
                samples = samples.numpy()
                labels = labels.numpy()
            # If datasets are already initialized, just append originals and labels
            if datasets_created:
                file["original"].resize(row_count + samples.shape[0], axis=0)
                file["original"][row_count:] = samples
                file["labels"].resize(row_count + labels.shape[0], axis=0)
                file["labels"][row_count:] = labels
            # Initialize datasets if necessary
            else:
                # Shape is now known, create datasets
                batch_size = samples.shape[0]
                file.create_dataset("original", shape=samples.shape, maxshape=(None,) + samples.shape[1:],
                                    chunks=samples.shape, dtype=samples.dtype)
                file.create_dataset("labels", shape=labels.shape, maxshape=(None,),
                                    chunks=labels.shape, dtype=labels.dtype)
                file["original"][:] = samples
                file["labels"][:] = labels
                # Each perturbation level has a separate HDF5 dataset inside the file
                for i, l in enumerate(perturbation_levels):
                    file.create_dataset(f"level_{i}", shape=samples.shape,
                                        maxshape=(None,) + samples.shape[1:],
                                        chunks=samples.shape,
                                        dtype=samples.dtype)
                datasets_created = True
            tries = 0
            batch_ok = False
            batch = []
            # Try to create a perturbed batch that doesn't change the model output
            while tries < max_tries and not batch_ok:
                tries += 1
                batch = []
                # Perturb and check if output hasn't changed
                for p_l in perturbation_levels:
                    perturbed_samples = p_fns[perturbation_fn](samples, p_l)
                    predictions = model.predict(torch.tensor(perturbed_samples))
                    batch_ok = not torch.any(predictions.argmax(axis=1) != torch.tensor(labels))
                    batch.append(perturbed_samples)
            if batch_ok:
                # Append each level of the batch to its corresponding HDF5 dataset
                for l, l_batch in enumerate(batch):
                    file[f"level_{l}"].resize(row_count + batch_size, axis=0)
                    file[f"level_{l}"][row_count:] = l_batch
            else:
                raise Exception("Maximum number of tries exceeded. Aborting.")
            row_count += batch_size
        # Save metadata
        metadata = {
            "perturbation_fn": perturbation_fn,
            "perturbation_levels": list(perturbation_levels)
        }
        json.dump(metadata, open(metadata_filename, "w"))
        return PerturbedImageDataset(data_location, name, batch_size)

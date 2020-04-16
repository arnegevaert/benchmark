from typing import Iterable, List, Callable
from os import path
import numpy as np
import os
import h5py
import torch
import warnings
import json


class NoisePerturbedDataset:
    def __init__(self, location):
        self.filename = path.join(location, "dataset.hdf5")
        assert(path.exists(self.filename))
        metadata = json.load(open(path.join(location, "meta.json")))
        self.perturbation_levels = metadata["perturbation_levels"]
        self.batch_size = metadata["batch_size"]
        self.sample_shape = metadata["sample_shape"]
        self.n_batches = metadata["n_batches"]
        self.file = h5py.File(self.filename, "r")

    def __iter__(self):
        count = 0
        while count + self.batch_size <= self.file["original"].shape[0]:
            yield {
                "perturbed": [self.file[f"level_{l}"][count:count+self.batch_size, :]
                              for l in range(len(self.perturbation_levels))],
                "original": self.file[f"original"][count:count+self.batch_size, :],
                "labels": self.file[f"labels"][count:count+self.batch_size]
            }
            count += self.batch_size


def generate_noise_perturbed_dataset(data: Iterable, location: str, perturbation_levels: List,
                                     max_tries: int, n_batches: int, model: Callable[[np.ndarray], np.ndarray] = None,
                                     normalization_fn: Callable[[np.ndarray], np.ndarray] = None):
    iterator = iter(data)
    # [batch_size, *sample_shape], [batch_size]
    batch, labels = next(iterator, None)
    batch_size, sample_shape = batch.shape[0], batch.shape[1:]
    # Construct filenames and create directories if necessary
    filename = path.join(location, "dataset.hdf5")
    metadata_filename = path.join(location, "meta.json")
    if not path.exists(filename):
        os.makedirs(location)
    file = h5py.File(filename, "w")

    # Create datasets in h5 file and allocate room for first batch
    # Dataset containing original samples
    file.create_dataset("original", shape=batch.shape, maxshape=(None,) + sample_shape,
                        chunks=batch.shape, dtype=batch.dtype)
    # Dataset containing labels of original samples
    file.create_dataset("labels", shape=labels.shape, maxshape=(None,), chunks=(batch_size,), dtype=batch.dtype)
    for l in range(len(perturbation_levels)):
        # A new dataset for every noise level
        file.create_dataset(f"level_{l}", shape=batch.shape, maxshape=(None,) + sample_shape,
                            chunks=batch.shape, dtype=batch.dtype)

    row_count, successful_batches = 0, 0
    while successful_batches < n_batches and batch is not None:
        tries, batch_ok, perturbed_batch = 0, False, []
        # Try to make a batch until it works or until maximum number of tries is reached
        while tries < max_tries and not batch_ok:
            batch_ok, perturbed_batch = True, []
            tries += 1
            for p_l in perturbation_levels:
                perturbed_samples = batch + (np.random.rand(*batch.shape)) * p_l
                if normalization_fn:
                    perturbed_samples = normalization_fn(perturbed_samples)
                predictions = model(perturbed_samples)
                batch_ok = batch_ok and not np.any(predictions.argmax(axis=1) != labels)
                if not batch_ok:
                    break
                perturbed_batch.append(perturbed_samples)
        if batch_ok:
            # Resize datasets to accommodate for new batch
            file["original"].resize(row_count + batch_size, axis=0)
            file["labels"].resize(row_count + batch_size, axis=0)
            for level, l_batch in enumerate(perturbed_batch):
                file[f"level_{level}"].resize(row_count + batch_size, axis=0)

            # Add batch to datasets
            file["original"][row_count:] = batch
            file["labels"][row_count:] = labels
            for i, l_batch in enumerate(perturbed_batch):
                file[f"level_{i}"][row_count:] = l_batch
            successful_batches += 1
            row_count += batch_size
            print(f"Batch successful: {successful_batches}/{n_batches}")
        else:
            print("Skipped batch.")
        batch, labels = next(iterator, (None, None))
    if successful_batches < n_batches:
        warnings.warn(f"Only {successful_batches}/{n_batches} were successful.")
    else:
        print(f"Successfully generated {successful_batches} batches.")
    metadata = {
        "perturbation_levels": perturbation_levels,
        "batch_size": batch_size,
        "sample_shape": sample_shape,
        "n_batches": successful_batches
    }
    json.dump(metadata, open(metadata_filename, "w"))
    file.flush()
    return NoisePerturbedDataset(location)

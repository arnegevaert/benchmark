import h5py
import torch
import json
import numpy as np
import warnings
import os
from os import path

from models import Model
from datasets import DerivedDataset


class NoisePerturbedDataset(DerivedDataset):
    def __init__(self, data_location, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.data_location = data_location
        self.filename = path.join(data_location, name, "dataset.hdf5")
        if not path.exists(self.filename):
            raise FileNotFoundError("Dataset not found, has it been generated?")
        self.metadata_filename = path.join(data_location, name, "meta.json")
        self.file = h5py.File(self.filename, 'r')
        self.metadata = json.load(open(self.metadata_filename))

    def __iter__(self):
        count = 0
        while count + self.batch_size <= self.file["original"].shape[0]:
            yield {
                "perturbed": [self.file[f"level_{l}"][count:count+self.batch_size, :]
                              for l in range(len(self.get_levels()))],
                "original": self.file[f"original"][count:count+self.batch_size, :],
                "labels": self.file[f"labels"][count:count+self.batch_size]
            }
            count += self.batch_size

    def get_levels(self):
        return self.metadata["perturbation_levels"]

    def get_fn(self):
        return self.metadata["perturbation_fn"]

    @staticmethod
    def generate(data_location, name, dataset: Dataset, model: Model,
                 perturbation_fn="noise", perturbation_levels=np.linspace(0, 1, 10), max_tries=10,
                 n_batches=64):
        assert(perturbation_fn in ["noise", "mean_shift"])
        p_fns = {
            "noise": lambda s, l: s + (np.random.rand(*s.shape)) * l,
            "mean_shift": lambda s, l: s + l
        }
        data_iterator = iter(dataset.get_test_data())
        filename = path.join(data_location, name, "dataset.hdf5")
        metadata_filename = path.join(data_location, name, "meta.json")
        if not path.exists(filename):
            os.makedirs(path.join(data_location, name))
        file = h5py.File(filename, 'w')
        datasets_created = False
        row_count = 0
        batch_size = None
        successful_batches = 0
        while successful_batches < n_batches:
            batch = next(data_iterator, None)
            if not batch:
                break
            # [batch_size, *sample_shape], [batch_size]
            samples, labels = batch
            batch_size = samples.shape[0]
            channels = samples.shape[1]
            # Convert to numpy if necessary
            if type(samples) == torch.Tensor:
                samples = samples.numpy()
                labels = labels.numpy()
            tries = 0
            batch_ok = False
            batch = []
            # Try to create a perturbed batch that doesn't change the model output
            while tries < max_tries and not batch_ok:
                batch_ok = True
                tries += 1
                batch = []
                # Perturb and check if output hasn't changed
                for p_l in perturbation_levels:
                    perturbed_samples = p_fns[perturbation_fn](samples, p_l)

                    im_mean = perturbed_samples.reshape((batch_size, channels, -1)).mean(2)\
                        .reshape((batch_size, channels, 1, 1))
                    im_std = perturbed_samples.reshape((batch_size, channels, -1)).std(2)\
                        .reshape((batch_size, channels, 1, 1))
                    perturbed_samples = (perturbed_samples - im_mean) / im_std

                    predictions = model.predict(torch.tensor(perturbed_samples))
                    batch_ok = batch_ok and not torch.any(predictions.argmax(axis=1) != torch.tensor(labels))
                    batch.append(perturbed_samples)
            if batch_ok:
                if datasets_created:
                    # Resize datasets to accommodate for new batch
                    file["original"].resize(row_count + batch_size, axis=0)
                    file["labels"].resize(row_count + batch_size, axis=0)
                    for l, l_batch in enumerate(batch):
                        file[f"level_{l}"].resize(row_count + batch_size, axis=0)
                else:
                    # Create datasets and allocate room for first batch
                    file.create_dataset("original", shape=samples.shape, maxshape=(None,) + samples.shape[1:],
                                        chunks=samples.shape, dtype=samples.dtype)
                    file.create_dataset("labels", shape=labels.shape, maxshape=(None,),
                                        chunks=labels.shape, dtype=labels.dtype)
                    for i, l_batch in enumerate(batch):
                        file.create_dataset(f"level_{i}", shape=samples.shape,
                                            maxshape=(None,) + samples.shape[1:],
                                            chunks=samples.shape,
                                            dtype=samples.dtype)
                    datasets_created = True

                # Add batch to datasets
                file["original"][row_count:] = samples
                file["labels"][row_count:] = labels
                for i, l_batch in enumerate(batch):
                    file[f"level_{i}"][row_count:] = l_batch
                successful_batches += 1
                row_count += batch_size
                print(f"Batch successful: {successful_batches}/{n_batches}")
            else:
                print("Skipped batch.")
        # Save metadata
        if successful_batches < n_batches:
            warnings.warn(f"Only {successful_batches}/{n_batches} were successful.")
        else:
            print(f"Successfully generated {successful_batches} batches.")
        metadata = {
            "perturbation_fn": perturbation_fn,
            "perturbation_levels": list(perturbation_levels)
        }
        json.dump(metadata, open(metadata_filename, "w"))
        file.flush()
        return NoisePerturbedDataset(data_location, name, batch_size)

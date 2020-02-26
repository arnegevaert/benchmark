import torch
import numpy as np
from models import Model
from datasets import ImageDataset


# TODO unnecessary conversion back and forth between NP and PT
def perturb_dataset(dataset: ImageDataset, model: Model, n_batches,
                    perturbation_fn="noise", perturbation_levels=np.linspace(0, 1, 10), max_tries=5):
    if perturbation_fn not in ["noise", "mean_shift"]:
        raise Exception("perturbation_fn must be noise or mean_shift")
    p_fns = {
        "noise": lambda s, l: s + (np.random.rand(*s.shape)) * l,
        "mean_shift": lambda s, l: s + l
    }
    perturbed = []
    labels = []
    originals = []
    for b in range(n_batches):
        tries = 0
        batch = []
        batch_ok = False
        while tries < max_tries and not batch_ok:
            tries += 1
            batch = []
            # [batch_size, **sample_shape], [batch_size]
            b_samples, b_labels = next(iter(dataset.get_test_loader()))
            for p_l in perturbation_levels:
                perturbed_samples = p_fns[perturbation_fn](b_samples, p_l)
                predictions = model.predict(perturbed_samples)
                batch_ok = not torch.any(predictions.argmax(axis=1) != b_labels)
                # [1, batch_size, *sample_shape]
                batch.append(perturbed_samples.unsqueeze(0))
        if not batch_ok:
            raise Exception("Maximum number of tries exceeded. Aborting.")
        batch = np.concatenate(batch, axis=0)  # [shift_levels, batch_size, *sample_shape]
        perturbed.append(np.expand_dims(batch, 1))  # [shift_levels, 1, batch_size, *sample_shape]
        labels.append(b_labels)
        originals.append(b_samples.unsqueeze(0))
    # [n_batches, batch_size, *sample_shape]
    originals = np.vstack(originals)
    # [n_batches, batch_size]
    labels = np.vstack(labels)
    # [noise_levels, n_batches, batch_size, *sample_shape]
    perturbed = np.concatenate(perturbed, axis=1)
    return torch.tensor(originals), torch.tensor(perturbed), torch.tensor(labels)


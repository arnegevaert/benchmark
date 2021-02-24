import torch
from typing import Callable
import random
from skimage.segmentation import slic
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PerturbationDataset(Dataset):
    def __init__(self, samples: np.ndarray, perturbation_size, num_perturbations):
        self.samples = samples
        self.perturbation_size = perturbation_size
        self.num_perturbations = num_perturbations

    def __len__(self):
        return self.num_perturbations

    def __getitem__(self, item):
        raise NotImplementedError


class GaussianPerturbation(PerturbationDataset):
    # perturbation_size is stdev of noise
    def __getitem__(self, item):
        perturbation_vector = np.random.normal(0, self.perturbation_size, self.samples.shape)
        perturbed_samples = self.samples - perturbation_vector
        return perturbed_samples, perturbation_vector


class SquareRemovalPerturbation(PerturbationDataset):
    # perturbation_size is (square height)/(image height)
    def __getitem__(self, item):
        height = self.samples.shape[2]
        width = self.samples.shape[3]
        square_size_int = int(self.perturbation_size * height)
        x_loc = random.randint(0, width - square_size_int)
        y_loc = random.randint(0, height - square_size_int)
        perturbation_mask = np.zeros(self.samples.shape)
        perturbation_mask[:, :, x_loc:x_loc + square_size_int, y_loc:y_loc + square_size_int] = 1
        perturbation_vector = self.samples * perturbation_mask
        perturbed_samples = self.samples - perturbation_vector
        return perturbed_samples, perturbation_vector


class SegmentRemovalPerturbation(PerturbationDataset):
    # perturbation size is number of segments
    def __init__(self, samples, perturbation_size, num_perturbations):
        super().__init__(samples, perturbation_size, num_perturbations)
        seg_samples = np.stack([slic(np.transpose(samples[i, ...], (1, 2, 0)),
                                     start_label=0, slic_zero=True)
                               for i in range(samples.shape[0])])
        self.seg_samples = np.expand_dims(seg_samples, axis=1)

    def __getitem__(self, item):
        perturbed_samples, perturbation_vectors = [], []
        # This needs to happen per sample, since samples don't necessarily have
        # the same number of segments
        for i in range(self.samples.shape[0]):
            seg_sample = self.seg_samples[i, ...]
            sample = self.samples[i, ...]
            # Get all segment numbers
            all_segments = np.unique(seg_sample)
            # Select segments to mask
            segments_to_mask = np.random.choice(all_segments, self.perturbation_size, replace=False)
            # Create boolean mask of pixels that need to be removed
            to_remove = np.isin(seg_sample, segments_to_mask)
            # Create perturbation vector by multiplying mask with image
            perturbation_vector = sample * to_remove.astype(np.float)
            perturbed_samples.append((sample - perturbation_vector).astype(np.float))
            perturbation_vectors.append(perturbation_vector)
        return np.stack(perturbed_samples, axis=0), np.stack(perturbation_vectors, axis=0)


_PERTURBATION_CLASSES = {
    "gaussian": GaussianPerturbation,
    "square": SquareRemovalPerturbation,
    "segment": SegmentRemovalPerturbation
}


def infidelity(samples: torch.Tensor, labels: torch.Tensor, model: Callable, attrs: torch.Tensor,
               perturbation_mode: str, perturbation_size: float, num_perturbations: int,
               writer=None):
    device = samples.device
    if perturbation_mode not in _PERTURBATION_CLASSES.keys():
        raise ValueError(f"Invalid perturbation mode {perturbation_mode}. "
                         f"Valid options are {', '.join(list(_PERTURBATION_CLASSES.keys()))}")
    perturbation_ds = _PERTURBATION_CLASSES[perturbation_mode](samples.cpu().numpy(),
                                                               perturbation_size, num_perturbations)
    perturbation_dl = DataLoader(perturbation_ds, batch_size=1, num_workers=4)

    # Get original model output
    with torch.no_grad():
        orig_output = (model(samples)).gather(dim=1, index=labels.unsqueeze(-1))  # [batch_size, 1]
    attrs = attrs.to(samples.device)
    # Replicate attributions along channel dimension if necessary (if explanation has fewer channels than image)
    if attrs.shape[1] != samples.shape[1]:
        shape = [1 for _ in range(len(attrs.shape))]
        shape[1] = samples.shape[1]
        attrs = attrs.repeat(*tuple(shape))
    attrs_flattened = attrs.flatten(1)

    infid = []
    for i_pert, (perturbed_samples, perturbation_vector) in enumerate(perturbation_dl):
        print(i_pert)
        # Get perturbation vector I and perturbed samples (x - I)
        #perturbed_samples = torch.tensor(perturbed_samples[0], dtype=torch.float, device=device)
        #perturbation_vector = torch.tensor(perturbation_vector[0], dtype=torch.float, device=device)
        perturbed_samples = perturbed_samples[0].float().to(device)
        perturbation_vector = perturbation_vector[0].float().to(device)
        if writer:
            writer.add_images("perturbation_vector", perturbation_vector, global_step=i_pert)
            writer.add_images("perturbed_samples", perturbed_samples, global_step=i_pert)

        with torch.no_grad():
            perturbed_output = model(perturbed_samples).gather(dim=1, index=labels.unsqueeze(-1))
        # Calculate dot product between each sample and its corresponding perturbation vector
        # This is equivalent to diagonal of matmul
        dot_product = (attrs_flattened * perturbation_vector.flatten(1)).sum(dim=1, keepdim=True)  # [batch_size, 1]
        pred_diff = orig_output - perturbed_output
        infid.append(((dot_product - pred_diff)**2).detach())
    # Take average across all perturbations
    infid = torch.cat(infid, dim=1).mean(dim=1)  # [batch_size]
    infid = infid.unsqueeze(1)  # [batch_size, 1]

    return infid.cpu()

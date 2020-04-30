import torch
import numpy as np
from sklearn.metrics import jaccard_score
import random
from typing import Iterable, Callable, Dict, Tuple


def impact_coverage(data: Iterable, sample_shape: Tuple, patch, target_label: int, patch_size_percent: float, model: Callable, methods: Dict[str, Callable],
                    device: str = "cpu"):
    image_size = sample_shape[-1]
    patch_size = int(((image_size**2)*patch_size_percent)**0.5)
    keep_list = []  # boolean mask of images to keep (not predicted as target class and successfully attacked)
    patch_masks = []  # boolean mask of location of patch
    critical_factor_mask = {m_name: [] for m_name in methods}  # boolean mask of location top factors

    for b, (samples, labels) in enumerate(data):
        samples, labels = torch.tensor(samples), torch.tensor(labels)
        samples = samples.to(device, non_blocking=True)
        labels = labels.numpy()
        predictions = model(samples).detach().cpu().numpy()
        # patch location same for all images in batch
        ind = random.randint(0, image_size - patch_size)
        samples[:, :, ind:ind + patch_size, ind:ind + patch_size] = patch
        patch_location_mask = np.zeros(samples.shape)
        patch_location_mask[:,:, ind: ind + patch_size, ind: ind + patch_size] = 1.
        patch_masks.append(patch_location_mask)
        adv_out = model(samples).detach().cpu().numpy()
        # use images that are not of the targeted class and are successfully attacked
        keep_indices = (predictions.argmax(axis=1) != target_label) * (adv_out.argmax(axis=1) == target_label) * (labels != target_label)
        keep_list.extend(keep_indices)
        for m_name in critical_factor_mask:
            attr_samples = samples.clone()
            method = methods[m_name]
            attrs = method(samples, target=target_label)
            attrs_shape = attrs.shape
            assert(len(attrs_shape) == 4) # shape[1] = 3 if colour channels are separate attributes, =1 pixel is attribute. might change later?
            attrs = attrs.reshape(attrs.shape[0], -1)
            sorted_indices = attrs.argsort().cpu()
            nr_top_attributes = attrs_shape[1]*patch_size**2
            to_mask = sorted_indices[:, -nr_top_attributes:]  # [batch_size, i]
            unraveled = np.unravel_index(to_mask, samples.shape[1:])
            batch_dim = np.column_stack([range(samples.shape[0]) for i in range(nr_top_attributes)])
            masks = np.zeros(attr_samples.shape)
            masks[(batch_dim, *unraveled)] = 1.
            critical_factor_mask[m_name].append(masks)
    result = {}
    patch_masks = np.vstack(patch_masks)[keep_list]
    for m_name in critical_factor_mask:
        cr_f_m = np.vstack(critical_factor_mask[m_name])
        cr_f_m = cr_f_m[keep_list]
        result[m_name] = jaccard_score(patch_masks.flatten(),cr_f_m.flatten())
    return result

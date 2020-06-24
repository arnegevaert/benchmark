from .masked_geometry_dataset import MaskedGeometryDataset
from typing import Dict, Callable
import itertools
import torch
import numpy as np


# TODO code duplication in model_masking_accuracy
def dataset_masking_accuracy(data: MaskedGeometryDataset, methods: Dict[str, Callable], n_batches=None):
    dl = data.get_dataloader(train=False, include_masks=True)
    n_batches = n_batches if n_batches else len(dl)
    iterator = itertools.islice(enumerate(dl), n_batches)
    jaccards = {m_name: [] for m_name in methods}
    for b, (samples, labels, masks) in iterator:
        print(f"Batch {b+1}/{n_batches}...")
        for m_name in methods:
            # Get attributions [batch_size, *sample_shape]
            attrs = methods[m_name](samples, labels)
            if len(attrs.shape) != 3:
                raise ValueError("Attributions must have shape [batch_size, rows, cols]")
            # Ignoring negative attributions, any feature is "important" if its attributions is > 0.01
            # TODO the way Jaccard indexes are being calculated should be configurable, create ROC curve
            attrs = (attrs > 0.).int()
            # Compute jaccard index of attrs with masks
            batch_size = samples.shape[0]
            card_intersect = torch.sum((attrs * masks).reshape((batch_size, -1)), dim=1)
            card_attrs = torch.sum(attrs.reshape((batch_size, -1)), dim=1)
            card_mask = torch.sum(masks.reshape(batch_size, -1), dim=1)
            jaccard = card_intersect / (card_attrs + card_mask - card_intersect)
            jaccards[m_name].append(jaccard)
    for m_name in methods:
        jaccards[m_name] = np.concatenate(jaccards[m_name])
    return jaccards

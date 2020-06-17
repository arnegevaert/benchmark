from .masked_dataset import MaskedDataset
from typing import Dict, Callable
import numpy as np
import itertools
import torch


def masking_accuracy(data: MaskedDataset, methods: Dict[str, Callable], n_batches=None):
    iterator = enumerate(data.get_dataloader(train=False))
    if n_batches:
        iterator = itertools.islice(iterator, n_batches)
    jaccards = {m_name: [] for m_name in methods}
    # shape of mask is [channels, rows, cols], take [0] to get [rows, cols] for aggregated attrs
    mask = data.get_mask()[0]
    for b, (samples, labels) in iterator:
        print(f"Batch {b+1}...")
        for m_name in methods:
            # Get attributions [batch_size, *sample_shape]
            attrs = methods[m_name](samples, labels)
            # Ignoring negative attributions, any feature is "important" if its attributions is > 0.01
            # TODO the way Jaccard indexes are being calculated should be configurable, create ROC curve
            attrs = (attrs > 0.01).int()
            # Compute jaccard index of attrs with mask
            card_intersect = torch.sum((attrs * mask).reshape((samples.shape[0], -1)), dim=1)
            card_attrs = torch.sum(attrs.reshape((attrs.shape[0], -1)), dim=1)
            card_mask = torch.sum(mask)
            jaccard = card_intersect / (card_attrs + card_mask - card_intersect)
            jaccards[m_name].append(jaccard)
    for m_name in methods:
        jaccards[m_name] = np.concatenate(jaccards[m_name])
    return jaccards

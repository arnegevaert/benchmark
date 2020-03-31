from models import MaskedCNN
from vars import DATASET_MODELS
from methods import get_method_constructors
import time
import torch
import numpy as np

# TODO this code only works for CIFAR10 now
DATASET = "CIFAR10"
BATCH_SIZE = 64
N_BATCHES = 16
MEDIAN_VALUE = -.788235
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution", "Random"]

model = MaskedCNN.load("../models/saved_models/cifar10_masked_cnn.pkl")

dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=True, download=False)
method_constructors = get_method_constructors(METHODS)
all_kwargs = {}
methods = {m_name: method_constructors[m_name](model, **all_kwargs.get(m_name, {})) for m_name in METHODS}

iterator = iter(dataset.get_test_data())
jaccards = {m_name: [] for m_name in METHODS}
for b in range(N_BATCHES):
    start_t = time.time()
    print(f"Batch {b + 1}/{N_BATCHES}...")
    samples, _ = next(iterator)
    output = model.predict(samples)
    # TODO these labels should be part of some synthetic dataset type
    labels = torch.all(model.mask(samples).reshape(BATCH_SIZE, -1) > MEDIAN_VALUE, dim=1).long()

    for m_name in METHODS:
        # Get attributions [BATCH_SIZE, channels, rows, cols]
        attrs = methods[m_name].attribute(samples, target=labels)
        attrs = (attrs > 0).long()
        # Compute jaccard index of attrs with mask
        mask = model.get_mask()
        card_intersect = (attrs * mask).reshape(BATCH_SIZE, -1).sum(dim=1)
        card_attrs_sq = (attrs**2).reshape(BATCH_SIZE, -1).sum(dim=1)
        card_mask_sq = (mask**2).reshape(BATCH_SIZE, -1).sum(dim=1)
        jaccard = card_intersect / (card_attrs_sq + card_mask_sq - card_intersect)
        jaccards[m_name].append(jaccard)

    end_t = time.time()
    seconds = end_t - start_t
    print(f"Batch {b+1}/{N_BATCHES} took {seconds:.2f}s. ETA: {seconds * (N_BATCHES-b-1):.2f}s.")

# TODO violin plots
for m_name in METHODS:
    avg_jaccard = torch.mean(torch.cat(jaccards[m_name]))
    print(f"Average jaccard index for {m_name}: {avg_jaccard}")

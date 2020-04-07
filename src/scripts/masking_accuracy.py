from models import MaskedCNN
from vars import DATASET_MODELS
from methods import get_method_constructors
import time
import torch
from lib import Report
import numpy as np
from scipy import stats

# TODO this code only works for CIFAR10 now
DATASET = "CIFAR10"
BATCH_SIZE = 64
N_BATCHES = 16
MEDIAN_VALUE = -.788235
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution", "Random"]  # , "Occlusion"]

model = MaskedCNN.load("../models/saved_models/cifar10_masked_cnn.pkl")

dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=False)
method_constructors = get_method_constructors(METHODS)
all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}
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

    # TODO calculation of jaccard index needs to be revised
    for m_name in METHODS:
        # Get attributions [BATCH_SIZE, channels, rows, cols]
        attrs = torch.abs(methods[m_name].attribute(samples, target=labels))
        # Compute jaccard index of attrs with mask
        mask = model.get_mask()

        card_intersect = (attrs * mask).detach().reshape(BATCH_SIZE, -1).sum(dim=1)
        card_attrs = (attrs**2).detach().reshape(BATCH_SIZE, -1).sum(dim=1)
        card_mask = (mask**2).sum()
        #jaccard = card_intersect / (card_attrs + card_mask - card_intersect)
        jaccard = card_intersect / card_mask
        jaccards[m_name].append(jaccard)

    end_t = time.time()
    seconds = end_t - start_t
    print(f"Batch {b+1}/{N_BATCHES} took {seconds:.2f}s. ETA: {seconds * (N_BATCHES-b-1):.2f}s.")

for m_name in METHODS:
    jaccards[m_name] = torch.cat(jaccards[m_name])
    print(stats.describe(jaccards[m_name]))

import pandas as pd
import seaborn as sns
data = pd.DataFrame.from_dict(jaccards)
sns.boxplot(data=data)

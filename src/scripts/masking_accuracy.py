from models import MaskedCNN
from vars import DATASET_MODELS
from methods import get_method_constructors
import time
import torch
from lib import Report
import numpy as np

# TODO this code only works for CIFAR10 now
DATASET = "CIFAR10"
BATCH_SIZE = 64
N_BATCHES = 16
MEDIAN_VALUE = -.788235
METHODS = ["GuidedGradCAM", "Gradient", "InputXGradient", "IntegratedGradients",
           "GuidedBackprop", "Deconvolution", "Random"]  # , "Occlusion"]
threshold_range = np.arange(0, 0.05, 0.005)

model = MaskedCNN.load("../models/saved_models/cifar10_masked_cnn.pkl")

dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=False)
method_constructors = get_method_constructors(METHODS)
all_kwargs = {"Occlusion": {"sliding_window_shapes": (1, 1, 1)}}
methods = {m_name: method_constructors[m_name](model, **all_kwargs.get(m_name, {})) for m_name in METHODS}

iterator = iter(dataset.get_test_data())
jaccards = {m_name: [[] for _ in threshold_range] for m_name in METHODS}
report = Report(METHODS)
for b in range(N_BATCHES):
    start_t = time.time()
    print(f"Batch {b + 1}/{N_BATCHES}...")
    samples, _ = next(iterator)
    output = model.predict(samples)
    # TODO these labels should be part of some synthetic dataset type
    labels = torch.all(model.mask(samples).reshape(BATCH_SIZE, -1) > MEDIAN_VALUE, dim=1).long()

    for m_name in METHODS:
        # Get attributions [BATCH_SIZE, channels, rows, cols]
        attrs = torch.abs(methods[m_name].attribute(samples, target=labels))
        # Compute jaccard index of attrs with mask
        mask = model.get_mask()
        for i, thresh in enumerate(threshold_range):
            thresh_attrs = ((attrs/attrs.max()) > (thresh*attrs.max())).long()
            card_intersect = (thresh_attrs * mask).detach().reshape(BATCH_SIZE, -1).sum(dim=1)
            card_attrs = thresh_attrs.detach().reshape(BATCH_SIZE, -1).sum(dim=1)
            card_mask = mask.sum()
            jaccard = card_intersect / (card_attrs + card_mask - card_intersect)
            jaccards[m_name][i].append(jaccard)

    end_t = time.time()
    seconds = end_t - start_t
    print(f"Batch {b+1}/{N_BATCHES} took {seconds:.2f}s. ETA: {seconds * (N_BATCHES-b-1):.2f}s.")

# TODO violin plots
for m_name in METHODS:
    m_avg_jaccards = []
    for i, thresh in enumerate(threshold_range):
        m_avg_jaccards.append(torch.cat(jaccards[m_name][i]).mean().item())
    report.add_summary_line(threshold_range, m_avg_jaccards, m_name)
report.render(x_label="threshold", y_label="avg jaccard index")


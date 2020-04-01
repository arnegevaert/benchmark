from models import MaskedCNN
from vars import DATASET_MODELS
from methods import get_method_constructors
import time
import torch
from scipy import stats
from lib import Report

# TODO this code only works for CIFAR10 now
DATASET = "CIFAR10"
BATCH_SIZE = 64
N_BATCHES = 1
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
worst_jaccards = {m_name: {"image": None, "attrs": None, "value": None} for m_name in METHODS}
best_jaccards = {m_name: {"image": None, "attrs": None, "value": None} for m_name in METHODS}
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
        attrs = methods[m_name].attribute(samples, target=labels)
        # Compute jaccard index of attrs with mask
        mask = model.get_mask()
        card_intersect = (attrs * mask).detach().reshape(BATCH_SIZE, -1).sum(dim=1)
        card_attrs_sq = (attrs**2).detach().reshape(BATCH_SIZE, -1).sum(dim=1)
        card_mask_sq = (mask**2).sum()
        jaccard = card_intersect / (card_attrs_sq + card_mask_sq - card_intersect)
        min_j, argmin_j = torch.min(jaccard, dim=0)
        max_j, argmax_j = torch.max(jaccard, dim=0)
        if not worst_jaccards[m_name]["value"] or min_j < worst_jaccards[m_name]["value"]:
            print(f"setting worst jaccards for {m_name}")
            worst_jaccards[m_name] = {
                "image": samples[argmin_j].detach().numpy(),
                "attrs": attrs[argmin_j].detach().numpy(),
                "value": min_j
            }
        if not best_jaccards[m_name]["value"] or max_j > best_jaccards[m_name]["value"]:
            print(f"setting best jaccards for {m_name}")
            best_jaccards[m_name] = {
                "image": samples[argmax_j].detach().numpy(),
                "attrs": attrs[argmax_j].detach().numpy(),
                "value": max_j
            }
        jaccards[m_name].append(jaccard)

    end_t = time.time()
    seconds = end_t - start_t
    print(f"Batch {b+1}/{N_BATCHES} took {seconds:.2f}s. ETA: {seconds * (N_BATCHES-b-1):.2f}s.")

# TODO violin plots
for m_name in METHODS:
    report.add_method_example_row(m_name, [worst_jaccards[m_name]["image"], worst_jaccards[m_name]["attrs"]])
    report.add_method_example_row(m_name, [best_jaccards[m_name]["image"], best_jaccards[m_name]["attrs"]])
    m_jaccards = torch.cat(jaccards[m_name]).detach().numpy()
    print(f"{m_name}:")
    print(stats.describe(m_jaccards))
    print()
report.render()


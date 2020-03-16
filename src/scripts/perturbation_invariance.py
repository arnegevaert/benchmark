from datasets import PerturbedImageDataset
from vars import DATASET_MODELS
from methods import get_all_method_constructors
import numpy as np
import torch
import matplotlib.pyplot as plt

GENERATE = False
DATA_ROOT = "../../data"
DATASET = "MNIST"
PERT_FN = "mean_shift"
MODEL = "CNN"
BATCH_SIZE = 4
N_BATCHES = 128

dataset_name = f"{DATASET}_{PERT_FN}"
dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
model = model_constructor()

method_constructors = get_all_method_constructors(include_random=False)

if GENERATE:
    dataset = dataset_constructor(batch_size=BATCH_SIZE, download=False, shuffle=True)
    iterator = iter(dataset.get_test_data())
    perturbed_dataset = PerturbedImageDataset.generate(DATA_ROOT, dataset_name, iterator, model,
                                                       perturbation_fn=PERT_FN,
                                                       perturbation_levels=np.linspace(0, 2, 10),
                                                       n_batches=N_BATCHES)
else:
    perturbed_dataset = PerturbedImageDataset(DATA_ROOT, dataset_name, BATCH_SIZE)

fig = plt.figure()
ax = plt.axes()
for key in method_constructors:
    print(f"Calculating for {key}...")
    method = method_constructors[key](model)
    diffs = [[] for _ in range(len(perturbed_dataset.get_levels()))]
    for b, b_dict in enumerate(perturbed_dataset):
        print(f"Batch {b+1}/{N_BATCHES}")
        orig = torch.tensor(b_dict["original"])
        labels = torch.tensor(b_dict["labels"])
        orig_attr = method.attribute(orig, target=labels).detach()
        for n_l, noise_level_batch in enumerate(b_dict["perturbed"]):
            noise_level_batch = torch.tensor(noise_level_batch)
            perturbed_attr = method.attribute(noise_level_batch, target=labels).detach()
            # [batch_size]
            avg_diff = np.average(np.abs(orig_attr - perturbed_attr))
            diffs[n_l].append(avg_diff)
    diffs = np.array(diffs)
    avg_diffs = np.average(diffs, axis=1)

    ax.plot(perturbed_dataset.get_levels(), diffs.mean(axis=1), label=key)
ax.legend()

from datasets import PerturbedImageDataset, MNIST
from models import MNISTCNN
from itertools import islice
from methods import Gradient, InputXGradient, IntegratedGradients
import numpy as np
import torch
import matplotlib.pyplot as plt

GENERATE = False
DATA_ROOT = "../../data"
DATASET = "MNIST_noise"
BATCH_SIZE = 4
N_BATCHES = 64

model = MNISTCNN()

methods = {
    "Gradient": Gradient(model.net),
    "InputXGradient": InputXGradient(model.net),
    "IntegratedGradients": IntegratedGradients(model.net)
}

if GENERATE:
    dataset = MNIST(batch_size=BATCH_SIZE, download=False)
    iterator = iter(dataset.get_test_loader())
    perturbed_dataset = PerturbedImageDataset.generate("../../data", "MNIST_noise", iterator, model,
                                                       perturbation_fn="noise",
                                                       perturbation_levels=np.linspace(0, 2, 10),
                                                       n_batches=N_BATCHES)
else:
    perturbed_dataset = PerturbedImageDataset(DATA_ROOT, DATASET, BATCH_SIZE)

fig = plt.figure()
ax = plt.axes()
for key in methods:
    print(f"Calculating for {key}...")
    method = methods[key]
    diffs = [[] for _ in range(len(perturbed_dataset.get_levels()))]
    for b, b_dict in enumerate(perturbed_dataset):
        print(f"Batch {b}/{N_BATCHES}")
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

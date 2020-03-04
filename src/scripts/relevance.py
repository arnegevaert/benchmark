from models import MNISTCNN, Model
from datasets import MNIST
from methods import Gradient, InputXGradient, IntegratedGradients, Random, Method
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
import torch


def relevance(data_loader: Iterable, model: Model, method: Method,
              n_pixels, n_batches=1, mask_value=0):
    result = []
    iterator = iter(data_loader)
    for b in range(n_batches):
        print(f"Batch {b+1}/{n_batches}")
        samples, labels = next(iterator)
        batch_size = samples.shape[0]
        print(labels)
        batch_result = []
        # [batch_size, *sample_shape]
        attrs = method.attribute(samples, target=labels)
        # Flatten each sample in order to get argmax per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        for i in range(n_pixels):
            # Get maximum attribution index for each image
            amax = torch.argmax(attrs, 1)  # [batch_size]
            # Mask value in images
            # unravel_index converts flat index (indices) to given shape
            unraveled_amax = np.unravel_index(amax, samples.shape[1:])
            samples[(range(batch_size), *unraveled_amax)] = mask_value
            # Set original attribution values to 0
            attrs[(range(batch_size), amax)] = 0
            # Get model output on masked image
            batch_result.append(model.predict(samples).gather(1, labels.reshape(-1, 1)))
        batch_result = torch.cat(batch_result, 1)  # [batch_size, n_pixels]
        result.append(batch_result)
    return torch.cat(result, 0).detach()  # [n_batches*batch_size, n_pixels]


DATA_ROOT = "../../data"
DATASET = "MNIST"
BATCH_SIZE = 64
N_BATCHES = 4
N_PIXELS = 256

model = MNISTCNN()
dataset = MNIST(batch_size=BATCH_SIZE, shuffle=False, download=False)

methods = {
    "Gradient": Gradient(model.net),
    "InputXGradient": InputXGradient(model.net),
    "IntegratedGradients": IntegratedGradients(model.net),
    "Random": Random()
}

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, N_PIXELS, N_PIXELS)
for key in methods:
    #res = relevance(dataset.get_test_data(), model, methods[key],
    #                n_pixels=N_PIXELS, n_batches=N_BATCHES, mask_value=dataset.mask_value).mean(dim=0)
    result = []
    iterator = iter(dataset.get_test_data())
    for b in range(N_BATCHES):
        print(f"Batch {b+1}/{N_BATCHES}")
        samples, labels = next(iterator)
        print(labels)
        batch_result = []
        # [batch_size, *sample_shape]
        attrs = methods[key].attribute(samples, target=labels)
        # Flatten each sample in order to get argmax per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        for i in range(N_PIXELS):
            # Get maximum attribution index for each image
            amax = torch.argmax(attrs, 1)  # [batch_size]
            # Mask value in images
            # unravel_index converts flat index (indices) to given shape
            unraveled_amax = np.unravel_index(amax, samples.shape[1:])
            samples[(range(BATCH_SIZE), *unraveled_amax)] = dataset.mask_value
            # Set original attribution values to 0
            attrs[(range(BATCH_SIZE), amax)] = 0
            # Get model output on masked image
            batch_result.append(model.predict(samples).gather(1, labels.reshape(-1, 1)))
        batch_result = torch.cat(batch_result, 1)  # [batch_size, n_pixels]
        result.append(batch_result)
    result = torch.cat(result, 0).detach().mean(dim=0)  # [n_batches*batch_size, n_pixels]
    ax.plot(x, result, label=key)
ax.set_xlabel("Number of masked pixels")
ax.set_ylabel("Classifier confidence")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True)

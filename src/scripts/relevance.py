from methods import get_all_method_constructors
from vars import DATASET_MODELS
import matplotlib.pyplot as plt
import numpy as np
import torch


def mask_most_important(x, attr):
    pass


DATA_ROOT = "../../data"
DATASET = "CIFAR10"
DOWNLOAD_DATASET = False
MODEL = "resnet20"
BATCH_SIZE = 32
N_BATCHES = 4
N_PIXELS = 128
PIXEL_STEP = 1

dataset_constructor = DATASET_MODELS[DATASET]["constructor"]
model_constructor = DATASET_MODELS[DATASET]["models"][MODEL]
method_constructors = get_all_method_constructors()

model = model_constructor()
dataset = dataset_constructor(batch_size=BATCH_SIZE, shuffle=False, download=DOWNLOAD_DATASET)

fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, N_PIXELS, N_PIXELS//PIXEL_STEP)
for key in method_constructors:
    print(f"Method: {key}")
    result = []
    iterator = iter(dataset.get_test_data())
    method = method_constructors[key](model)
    for b in range(N_BATCHES):
        print(f"Batch {b+1}/{N_BATCHES}")
        samples, labels = next(iterator)
        batch_result = []
        # [batch_size, *sample_shape]
        attrs = method.attribute(samples, target=labels)
        # Flatten each sample in order to get argmax per sample
        attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, -1]
        for i in range(0, N_PIXELS, PIXEL_STEP):
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

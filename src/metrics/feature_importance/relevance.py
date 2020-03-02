from models import Model
from methods import Method
from torch.utils.data import DataLoader
import torch


def relevance(data_loader: DataLoader, model: Model, method: Method,
              n_pixels, n_batches=1, mask_value=0):
    result = []
    iterator = iter(data_loader)
    for b in range(n_batches):
        print(f"Batch {b+1}/{n_batches}")
        samples, labels = next(iterator)
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
            samples[(range(data_loader.batch_size), *unraveled_amax)] = mask_value
            # Set original attribution values to 0
            attrs[(range(data_loader.batch_size), amax)] = 0
            # Get model output on masked image
            batch_result.append(model.predict(samples).gather(1, labels.reshape(-1, 1)))
        batch_result = torch.cat(batch_result, 1)  # [batch_size, n_pixels]
        result.append(batch_result)
    return torch.cat(result, 0).detach()  # [n_batches*batch_size, n_pixels]


if __name__ == '__main__':
    from models import MNISTCNN
    from datasets import MNIST
    from methods import Gradient, InputXGradient, IntegratedGradients, DeepLift, Random
    import matplotlib.pyplot as plt
    import numpy as np
    model = MNISTCNN()
    dataset = MNIST(batch_size=64, shuffle=False)
    methods = {
        "Gradient": Gradient(model.net),
        "InputXGradient": InputXGradient(model.net),
        "IntegratedGradients": IntegratedGradients(model.net),
        "DeepLift": DeepLift(model.net),
        "Random": Random()
    }
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 256, 256)
    for key in methods:
        res = relevance(dataset.get_test_loader(), model, methods[key],
                        n_pixels=256, n_batches=2, mask_value=dataset.mask_value).mean(dim=0)
        ax.plot(x, res, label=key)
    ax.set_xlabel("Number of masked pixels")
    ax.set_ylabel("Classifier confidence")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True)

import numpy as np
import warnings
from models import Model
from methods import Method
from datasets import ImageDataset


# TODO unnecessary conversion back and forth between NP and PT
def shift_dataset(dataset: ImageDataset, model: Model, n_batches,
                  shift_levels=np.linspace(-.1, .1, 10), max_tries=5):
    shifted = []
    labels = []
    originals = []
    for b in range(n_batches):
        tries = 0
        batch = []
        batch_ok = False
        while tries < max_tries and not batch_ok:
            tries += 1
            batch = []
            # [batch_size, **sample_shape], [batch_size]
            b_samples, b_labels = next(iter(dataset.get_test_loader()))
            for mean_shift in shift_levels:
                shifted_samples = b_samples + mean_shift
                predictions = model.predict(shifted_samples)
                batch_ok = not torch.any(predictions.argmax(axis=1) != b_labels)
                # [1, batch_size, *sample_shape]
                batch.append(shifted_samples.unsqueeze(0))
        if not batch_ok:
            warnings.warn("Maximum number of tries exceeded. Aborting.")
            return
        batch = np.concatenate(batch, axis=0)  # [shift_levels, batch_size, *sample_shape]
        shifted.append(np.expand_dims(batch, 1))  # [shift_levels, 1, batch_size, *sample_shape]
        labels.append(b_labels)
        originals.append(b_samples.unsqueeze(0))
    # [n_batches, batch_size, *sample_shape]
    originals = np.vstack(originals)
    # [n_batches, batch_size]
    labels = np.vstack(labels)
    # [shift_levels, n_batches, batch_size, *sample_shape]
    shifted = np.concatenate(shifted, axis=1)
    return torch.tensor(originals), torch.tensor(shifted), torch.tensor(labels)


def mean_shift_invariance(originals: np.ndarray, shifted: np.ndarray, labels: np.ndarray, method: Method):
    n_levels = shifted.shape[0]
    n_batches = shifted.shape[1]
    all_diffs = []
    for b in range(n_batches):
        print(f"Batch {b}/{n_batches}")
        # [batch_size, *sample_shape]
        orig_attr = method.attribute(originals[b, :], target=labels[b, :]).detach()
        diffs = []
        for l in range(n_levels):
            # [batch_size, *sample_shape]
            shifted_attr = method.attribute(shifted[l, b, :], target=labels[b, :]).detach()
            # [batch_size]
            attr_diff = np.abs(orig_attr - shifted_attr).reshape((orig_attr.shape[0], -1)).sum(axis=1)
            diffs.append(attr_diff)
        all_diffs.append(np.vstack(diffs))  # [n_levels, batch_size]
    return np.concatenate(all_diffs, axis=1)  # [n_levels, n_batches*batch_size]


if __name__ == '__main__':
    from models import MNISTCNN
    from datasets import MNIST
    from methods import *
    import matplotlib.pyplot as plt

    dataset = MNIST(batch_size=4, download=False)
    model = MNISTCNN(dataset=dataset)
    methods = {
        "Saliency": Saliency(model.net),
        "InputXGradient": InputXGradient(model.net),
        "IntegratedGradients": IntegratedGradients(model.net),
        "DeepLift": DeepLift(model.net)
    }

    shift_levels = np.linspace(-.1, .1, 11)
    originals, shifted, labels = shift_dataset(dataset, model, n_batches=64, shift_levels=shift_levels)
    fig = plt.figure()
    ax = plt.axes()
    for key in methods:
        print(f"Calculating Mean Shift Invariance for {key}...")
        diffs = mean_shift_invariance(originals, shifted, labels, methods[key])
        ax.plot(shift_levels, diffs.mean(axis=1), label=key)
    ax.legend()

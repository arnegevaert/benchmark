import torch
import numpy as np
from models import Model
from methods import Method
from datasets import ImageDataset
import warnings


def perturb_dataset(dataset: ImageDataset, model: Model, n_batches,
                    noise_levels=np.linspace(0, 1, 10), max_tries=5):
    perturbed = []
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
            for noise_level in noise_levels:
                perturbed_samples = b_samples + (np.random.rand(*b_samples.shape)) * noise_level
                predictions = model.predict(perturbed_samples)
                batch_ok = not torch.any(predictions.argmax(axis=1) != b_labels)
                # [1, batch_size, *sample_shape]
                batch.append(perturbed_samples.unsqueeze(0))
        if not batch_ok:
            raise Exception("Maximum number of tries exceeded. Aborting.")
        batch = np.concatenate(batch, axis=0)  # [shift_levels, batch_size, *sample_shape]
        perturbed.append(np.expand_dims(batch, 1))  # [shift_levels, 1, batch_size, *sample_shape]
        labels.append(b_labels)
        originals.append(b_samples.unsqueeze(0))
    # [n_batches, batch_size, *sample_shape]
    originals = np.vstack(originals)
    # [n_batches, batch_size]
    labels = np.vstack(labels)
    # [noise_levels, n_batches, batch_size, *sample_shape]
    perturbed = np.concatenate(perturbed, axis=1)
    return torch.tensor(originals), torch.tensor(perturbed), torch.tensor(labels)

# TODO we now use the average sum of differences (per sample). Consider using variance, or average difference per pixel?
def robustness(originals: np.ndarray, perturbed: np.ndarray, labels: np.ndarray, method: Method):
    n_levels = perturbed.shape[0]
    n_batches = perturbed.shape[1]
    all_diffs = []
    for b in range(n_batches):
        print(f"Batch {b}/{n_batches}")
        # [batch_size, *sample_shape]
        orig_attr = method.attribute(originals[b, :], target=labels[b, :]).detach()
        diffs = []
        for l in range(n_levels):
            # [batch_size, *sample_shape]
            shifted_attr = method.attribute(perturbed[l, b, :], target=labels[b, :]).detach()
            # [batch_size]
            attr_diff = np.abs(orig_attr - shifted_attr).reshape((orig_attr.shape[0], -1)).sum(axis=1)
            diffs.append(attr_diff)
        all_diffs.append(np.vstack(diffs))  # [n_levels, batch_size]
    return np.concatenate(all_diffs, axis=1)  # [n_levels, n_batches*batch_size]


class Robustness():
    def __init__(self, model: Model, method: Method, dataset: ImageDataset):
        self.model = model
        self.method = method
        self.dataset = dataset

    def measure(self, n_batches=1, noise_levels=np.linspace(0, 1, 10), n_perturbations=8):
        test_loader = self.dataset.get_test_loader()
        result = []
        for b in range(n_batches):
            print(f"Batch {b}/{n_batches}")
            batch_result = []
            # Get samples
            samples, labels = next(iter(test_loader))
            # Generate perturbations for each noise level
            for noise_level in noise_levels:
                # First verify data shape (must be [batch_size, 1, n_rows, n_cols])
                # TODO this will not work for datasets with other rank (e.g. color channels)
                if samples.shape[1] != 1:
                    samples.unsqueeze(1)
                # Duplicate images along second axis ([batch_size, n_perturbations, n_rows, n_cols])
                repeated_samples = samples.repeat((1, n_perturbations, 1, 1))
                # Add noise ([batch_size, n_perturbations, n_rows, n_cols])
                repeated_samples += (np.random.rand(*repeated_samples.shape)) * noise_level
                # merge first 2 dimensions ([batch_size*n_perturbations, n_rows, n_cols])
                orig_s = repeated_samples.shape
                repeated_samples = repeated_samples.view(-1, orig_s[2], orig_s[3])
                # Repeat each label n_perturbations times: [batch_size*n_perturbations]
                repeated_labels = torch.cat(n_perturbations*[labels.reshape(-1, 1)], 1).flatten()
                # Get attribution values ([batch_size*n_perturbations, n_rows, n_cols])
                # TODO if noise causes model to mispredict, this is not sound
                attrs = self.method.attribute(repeated_samples.unsqueeze(1), target=repeated_labels).detach().numpy()
                # Reshape back into original shape
                attrs = attrs.reshape(*orig_s)  # [batch_size, n_perturbations, n_rows, n_cols]
                # Get variance along second axis (perturbations)
                # TODO is this the right measure?
                variance = np.var(attrs, axis=1)  # [batch_size, n_rows, n_cols]
                # Get average of variance for each pixel
                avg_variance = np.mean(variance, axis=(1, 2))  # [batch_size]
                batch_result.append(avg_variance.reshape(-1, 1))
            batch_result = np.concatenate(batch_result, axis=1)  # [batch_size, noise_levels]
            result.append(batch_result)
        result = np.concatenate(result, axis=0)
        return result  # [n_batches*batch_size, noise_levels]


if __name__ == '__main__':
    import time
    from models import MNISTCNN
    from datasets import MNIST
    from methods import Saliency, InputXGradient, IntegratedGradients, DeepLift
    import matplotlib.pyplot as plt
    import numpy as np
    dataset = MNIST(batch_size=4, download=False)
    model = MNISTCNN(dataset=dataset)
    methods = {
        #"Saliency": Saliency(model.net),
        "InputXGradient": InputXGradient(model.net),
        #"IntegratedGradients": IntegratedGradients(model.net),
        "DeepLift": DeepLift(model.net)
    }
    noise_levels = np.linspace(0, 2, 10)
    originals, shifted, labels = perturb_dataset(dataset, model, n_batches=64, noise_levels=noise_levels)
    fig = plt.figure()
    ax = plt.axes()
    all_diffs = {}
    for key in methods:
        print(f"Calculating Robustness for {key}...")
        diffs = robustness(originals, shifted, labels, methods[key])
        all_diffs[key] = diffs
        ax.plot(noise_levels, diffs.mean(axis=1), label=key)
    ax.legend()

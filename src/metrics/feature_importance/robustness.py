import torch
import numpy as np
from metrics import Metric
from models import Model
from methods import Method
from datasets import ImageDataset


class Robustness(Metric):
    def __init__(self, model: Model, method: Method, dataset: ImageDataset):
        super().__init__(model, method)
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
    start_t = time.time()
    dataset = MNIST(batch_size=1, download=False)
    model = MNISTCNN(dataset=dataset)
    methods = {
        "Saliency": Saliency(model.net),
        "InputXGradient": InputXGradient(model.net),
        #"IntegratedGradients": IntegratedGradients(model.net),
        "DeepLift": DeepLift(model.net)
    }
    noise_levels = np.linspace(0, 2, 20)
    n_levels = len(noise_levels)
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, n_levels, n_levels)
    for key in methods:
        print(f"Calculating robustness for {key}...")
        metric = Robustness(model, methods[key], model.dataset)
        res = metric.measure(n_batches=64, noise_levels=noise_levels, n_perturbations=64).mean(axis=0)
        ax.plot(x, res, label=key)
    ax.legend()
    end_t = time.time()
    print(f"Ran for {end_t-start_t} seconds")

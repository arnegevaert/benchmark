import torch
import numpy as np
from metrics import Metric
from models import Model
from methods import Method
from datasets import ImageDataset


class MeanShiftInvariance(Metric):
    def __init__(self, model: Model, method: Method, dataset: ImageDataset):
        super().__init__(model, method)
        self.dataset = dataset

    def measure(self, n_batches=1, shift_levels=np.linspace(-1, 1, 10)):
        test_loader = self.dataset.get_test_loader()
        result = []
        misclassifications = 0
        for b in range(n_batches):
            print(f"Batch {b}/{n_batches}")
            batch_result = []
            samples, labels = next(iter(test_loader))  # [batch_size, 1, n_rows, n_cols], [batch_size]
            # Get attribution on original sample ([batch_size, 1, n_rows, n_cols])
            original_attrs = self.method.attribute(samples, target=labels).detach().numpy()
            for mean_shift in shift_levels:
                shifted_samples = samples + mean_shift
                predictions = self.model.predict(shifted_samples)
                misclassifications += torch.sum(predictions.argmax(axis=1) != labels)
                # TODO if mean shift causes model to mispredict, this is not sound
                attrs = self.method.attribute(shifted_samples, target=labels).detach().numpy()
                # [batch_size]
                attr_diff = np.abs(attrs - original_attrs).reshape((attrs.shape[0], -1)).sum(axis=1)
                batch_result.append(attr_diff.reshape(-1, 1))
            batch_result = np.concatenate(batch_result, axis=1)  # [batch_size, shift_levels]
            result.append(batch_result)
        result = np.concatenate(result, axis=0)
        if misclassifications > 0:
            print(f"WARNING: {misclassifications}/{result.shape[0]} samples were misclassified")
        return result  # [n_batches*batch_size, shift_levels]


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
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(-.1, .1, 10)
    for key in methods:
        print(f"Calculating Mean Shift Invariance for {key}...")
        metric = MeanShiftInvariance(model, methods[key], model.dataset)
        res = metric.measure(n_batches=64, shift_levels=x)
        ax.plot(x, res.mean(axis=0), label=key)
    ax.legend()

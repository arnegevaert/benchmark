from metrics import Metric
from models import Model
from methods import Method
from datasets import Dataset
import torch


class Relevance(Metric):
    def __init__(self, model: Model, method: Method, dataset: Dataset):
        super().__init__(model, method)
        self.dataset = dataset

    # TODO add random selection as baseline
    def measure(self, n_batches=1, n_pixels=256):
        test_loader = self.dataset.get_test_loader()
        result = []
        for b in range(n_batches):
            batch_result = []
            # Get samples
            samples, labels = next(iter(test_loader))
            orig_shape = samples.shape
            # Calculate attributions and flatten per sample
            attrs = self.method.attribute(samples, target=labels)  # [batch_size, 1, n_rows, n_cols]
            attrs = attrs.reshape(attrs.shape[0], -1)  # [batch_size, n_rows*n_cols]
            for i in range(n_pixels):
                # Get maximum attribution index for each image
                amax = torch.argmax(attrs, 1)  # [batch_size]
                # Mask value in images
                # TODO this is probably inefficient, index directly
                samples = samples.reshape(orig_shape[0], -1)  # [batch_size, n_rows*n_cols]
                samples[range(orig_shape[0]), amax] = self.dataset.mask_value
                samples = samples.reshape(*orig_shape)
                # Set attribution values to 0
                attrs[range(orig_shape[0]), amax] = 0.
                # Get model output on masked image
                batch_result.append(self.model.predict(samples).gather(1, labels.reshape(-1, 1)))
            batch_result = torch.cat(batch_result, 1)  # [batch_size, n_pixels]
            result.append(batch_result)
        return torch.cat(result, 0).detach().numpy()  # [n_batches*batch_size, n_pixels]


if __name__ == '__main__':
    from models import MNISTCNN
    from datasets import MNIST
    from methods import Saliency, InputXGradient, IntegratedGradients, DeepLift
    import matplotlib.pyplot as plt
    import numpy as np
    model = MNISTCNN()
    dataset = MNIST(batch_size=64)
    methods = {
        "Saliency": Saliency(model.net),
        "InputXGradient": InputXGradient(model.net),
        "IntegratedGradients": IntegratedGradients(model.net),
        "DeepLift": DeepLift(model.net)
    }
    fig = plt.figure()
    ax = plt.axes()
    x = np.linspace(0, 256, 256)
    for key in methods:
        metric = Relevance(model, methods[key], dataset)
        res = metric.measure(n_batches=2).mean(axis=0)
        ax.plot(x, res, label=key)
    ax.legend()

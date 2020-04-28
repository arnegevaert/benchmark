import torch.nn as nn
import torch
import torch.nn.functional as F
from benchmark.masking_accuracy import MaskedDataset
from typing import Dict, Callable
import numpy as np
import itertools


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def get_logits(self, x):
        relu = nn.ReLU()
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.dropout3(x)
        return self.fc3(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        return F.softmax(logits, dim=1)


def masking_accuracy(data: MaskedDataset, methods: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
                     n_batches=None):
    iterator = itertools.islice(enumerate(data.get_test_data()), n_batches) if n_batches else enumerate(data.get_test_data())
    jaccards = {m_name: [] for m_name in methods}
    mask = data.get_mask()
    for b, (samples, labels) in iterator:
        print(f"Batch {b+1}...")
        for m_name in methods:
            # Get attributions [batch_size, *sample_shape]
            attrs = methods[m_name](samples, labels)
            # Ignoring negative attributions, any feature is "important" if its attributions is > 0.01
            # TODO the way Jaccard indexes are being calculated should be configurable, create ROC curve
            attrs = (attrs > 0.01).astype(int)
            # Compute jaccard index of attrs with mask
            card_intersect = np.sum((attrs * mask).reshape((samples.shape[0], -1)), axis=1)
            card_attrs = np.sum(attrs.reshape((attrs.shape[0], -1)), axis=1)
            card_mask = np.sum(mask)
            jaccard = card_intersect / (card_attrs + card_mask - card_intersect)
            jaccards[m_name].append(jaccard)
    for m_name in methods:
        jaccards[m_name] = np.concatenate(jaccards[m_name])
    return jaccards


if __name__ == '__main__':
    from benchmark.masking_accuracy import MaskedNeuralNetwork, train_masked_network, test_masked_network, MaskedDataset
    from util.methods import get_method_constructors
    from util.datasets import Cifar
    import pickle as pkl
    from scipy import stats

    # Initialization
    DATASET = "CIFAR10"
    BATCH_SIZE = 64
    N_BATCHES = 16
    MEDIAN_VALUE = -.788235
    METHODS = ["Gradient", "InputXGradient", "IntegratedGradients",
               "GuidedBackprop", "Deconvolution", "Random"]  # , "Occlusion"]
    MODEL_LOC = "../../data/models/cifar10_masked_cnn.pkl"
    LOAD_MODEL = True

    # Create masked model and masked dataset
    orig_ds = Cifar(version="cifar10", batch_size=BATCH_SIZE, shuffle=False,
                    download=False, data_location="../../data/CIFAR10")
    masked_ds = MaskedDataset(orig_ds.get_train_loader(), orig_ds.get_test_loader(), radius=5,
                              mask_value=0., med_of_min=MEDIAN_VALUE)
    if LOAD_MODEL:
        with open(MODEL_LOC, "rb") as f:
            model = pkl.load(f)
        test_masked_network(model, masked_ds)
    else:
        model = MaskedNeuralNetwork(sample_shape=(3, 32, 32), mask_radius=5, mask_value=0., net=Net())
        # Train the model on synthetic labels
        train_masked_network(model, masked_ds, lr=1.0, gamma=0.7, epochs=2)
        with open(MODEL_LOC, "wb") as f:
            pkl.dump(model, f)

    # Get methods
    method_constructors = get_method_constructors(METHODS)
    methods = {m_name: method_constructors[m_name](model) for m_name in METHODS}

    results = masking_accuracy(masked_ds, methods)
    for key in results:
        print(f"{key}:")
        print(stats.describe(results[key]))

    import pandas as pd
    import seaborn as sns
    data = pd.DataFrame.from_dict(results)
    sns.boxplot(data=data)


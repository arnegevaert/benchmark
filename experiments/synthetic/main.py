from sklearn import datasets
from attrbench.evaluation.independent import *
from torch import nn, optim
import matplotlib.pyplot as plt


# https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html#sphx-glr-auto-examples-datasets-plot-random-dataset-py


if __name__ == "__main__":
    num_features = 2
    num_hidden = 50
    num_classes = 3

    model = nn.Sequential([
        nn.Linear(num_features, num_hidden),
        nn.ReLU(),
        nn.Linear(num_hidden, num_classes)
    ])

    X, y = datasets.make_multilabel_classification(
        n_samples=1000,
        n_features=num_features,
        n_classes=num_classes
    )

    num_epochs = 10
    optimizer = optim.Adam(model.parameters())
    for e in range(num_epochs):
        pass
from sklearn import datasets, model_selection, preprocessing
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm
import argparse
import os
from os import path
import numpy as np
import json


# https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html#sphx-glr-auto-examples-datasets-plot-random-dataset-py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--num-features", type=int)
    parser.add_argument("-i", "--num-informative", type=int)
    parser.add_argument("-c", "--num-classes", type=int)
    parser.add_argument("-n", "--num-nodes", type=int)
    parser.add_argument("-o", "--out-dir", type=str)
    args = parser.parse_args()

    # Create model and dataset
    model = nn.Sequential(
        nn.Linear(args.num_features, args.num_nodes),
        nn.ReLU(),
        nn.Linear(args.num_nodes, args.num_classes)
    )
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    X, y = datasets.make_classification(
        n_samples=10000,
        n_features=args.num_features,
        n_informative=args.num_informative,
        n_redundant=args.num_features - args.num_informative,
        n_classes=args.num_classes,
        n_clusters_per_class=1,
    )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle=True)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    ds_train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    dl_train = DataLoader(ds_train, batch_size=64)

    # Train on train set
    num_epochs = 10
    for e in tqdm(range(num_epochs)):
        num_correct, num_total = 0, 0
        for samples, labels in dl_train:
            optimizer.zero_grad()

            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            num_total += samples.size(0)
            num_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    print(f"Train accuracy: {num_correct / num_total:.2f}")

    # Test on test set
    ds_test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))
    dl_test = DataLoader(ds_test, batch_size=64)
    num_correct, num_total = 0, 0
    for samples, labels in dl_test:
        outputs = model(samples)
        num_total += samples.size(0)
        num_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    print(f"Test accuracy: {num_correct / num_total:.2f}")

    # Save model and data
    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    torch.save(model.state_dict(), path.join(args.out_dir, "model.pt"))
    np.save(path.join(args.out_dir, "x_train.npy"), X_train)
    np.save(path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(path.join(args.out_dir, "x_test.npy"), X_test)
    np.save(path.join(args.out_dir, "y_test.npy"), y_test)
    with open(path.join(args.out_dir, "args.json"), "w") as fp:
        json.dump(vars(args), fp)

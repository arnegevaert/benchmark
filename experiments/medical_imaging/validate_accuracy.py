import argparse
import torch
from experiments.medical_imaging.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sklearn.metrics as metrics




def test_epoch(model, dl):

    model.to(device)
    model.eval()
    with torch.no_grad():
        predictions=torch.tensor([])
        true_labels=[]
        for batch, labels in tqdm(dl):
            batch = batch.to(device)
            out = model(batch)
            predictions= torch.cat((predictions,out.cpu()),0)
            true_labels.extend(labels.numpy())

        true_labels = np.array(true_labels)

        # multi-label classification
        if len(true_labels.shape) >=2:
            predictions = torch.sigmoid(predictions).numpy()

            acc = metrics.accuracy_score(true_labels, predictions>0.5)
            balanced_acc = metrics.recall_score(true_labels, predictions>0.5, average='macro')
            auc = metrics.roc_auc_score(true_labels, predictions, average="macro", multi_class='ovr')
        # multi-class classification
        elif true_labels.max() !=1:
            predictions= torch.softmax(predictions,1).numpy()
            acc = metrics.accuracy_score(true_labels, predictions.argmax(axis=1))
            balanced_acc = metrics.balanced_accuracy_score(true_labels, predictions.argmax(axis=1))
            auc = metrics.roc_auc_score(true_labels,predictions,average="macro",multi_class='ovr')
        # binary classification
        else:
            if len(predictions.shape) ==2:
                # binary, but model has 2 separate outputs
                predictions = predictions[:,1]
            predictions = torch.sigmoid(predictions).numpy()
            acc = metrics.accuracy_score(true_labels, predictions>0.5)
            balanced_acc = metrics.balanced_accuracy_score(true_labels, predictions>0.5)
            auc = metrics.roc_auc_score(true_labels,predictions)

    return acc,balanced_acc, auc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")

    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # Get dataset, model, methods
    ds, model, sample_shape = get_dataset_model(args.dataset)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    acc,balanced_acc, auc = test_epoch(model, dl)
    print("validation set results:\n"
          "accuracy: {:f} \n"
          "balanced accuracy: {:f} \n"
          "AUC: {:f}".format(acc,balanced_acc,auc))

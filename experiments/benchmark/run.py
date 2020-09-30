import argparse
import itertools
from torch.utils.data import DataLoader
from attrbench.evaluation import *
from experiments.benchmark.util import get_ds_model_method, get_mask_range
import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    dataset, model, method = get_ds_model_method(args.dataset, args.model, args.method, args.batch_size)
    mask_range = get_mask_range(dataset)

    dataloader = itertools.islice(DataLoader(dataset, batch_size=args.batch_size), args.num_batches)
    for i, (batch, labels) in enumerate(dataloader):
        print(f"Batch {i + 1}/{args.num_batches}...")
        batch = batch.to(device)
        labels = labels.to(device)

        # Infidelity
        print("Infidelity...")
        batch_infidelity = infidelity(batch, labels, model, method,
                                      perturbation_range=list(np.linspace(.01, .3, 10)),
                                      num_perturbations=16)

        # Insertion curves
        print("Insertion curves...")
        batch_ins_curves = insertion_deletion_curves(batch, labels, model, method, mask_range,
                                                     mask_value=0., mode="insertion")

        # Deletion curves
        print("Deletion curves...")
        batch_del_curves = insertion_deletion_curves(batch, labels, model, method, mask_range,
                                                     mask_value=0., mode="deletion")

        # Max-sensitivity
        print("Max-sensitivity...")
        batch_max_sens = max_sensitivity(batch, labels, method,
                                         perturbation_range=list(np.linspace(.01, .3, 10)),
                                         num_perturbations=16)

        # Sensitivity-n
        print("Sensitivity-n...")
        batch_sens_n = sensitivity_n(batch, labels, model, method, mask_range,
                                     num_subsets=100, mask_value=0.)

        # Impact score (strict)
        print("Strict impact score...")
        batch_s_impact_score = impact_score(batch, labels, model, mask_range, method,
                                            mask_value=0., strict=False)

        # Impact score (non-strict)
        print("Non-strict impact score...")
        batch_impact_score = impact_score(batch, labels, model, mask_range, method,
                                          mask_value=0., strict=True)

        print(f"Batch {i + 1}/{args.num_batches} finished.")
        print()

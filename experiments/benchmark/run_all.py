import argparse
import itertools
from torch.utils.data import DataLoader
from attrbench.evaluation import *
from experiments.benchmark.util import get_ds_model_method, get_mask_range
import numpy as np
import torch
from tqdm import tqdm
from os import path
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--model", type=str, default="CNN")
    parser.add_argument("--method", type=str, default="Gradient")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--out-dir", type=str, default="out")
    # Parse arguments
    args = parser.parse_args()
    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # Get model, dataset, method, params
    dataset, model, method = get_ds_model_method(args.dataset, args.model, args.method, args.batch_size)
    model.to(device)
    model.eval()
    mask_range = get_mask_range(args.dataset)

    # Prepare results
    res = {}
    for name in ["infidelity", "insertion", "deletion", "max-sens", "sens-n"]:
        res[name] = []
    for name in ["impact", "s-impact"]:
        res[name] = {"counts": [], "total": 0}

    # Calculate metrics for each batch
    dataloader = itertools.islice(DataLoader(dataset, batch_size=args.batch_size), args.num_batches)
    for i, (batch, labels) in enumerate(dataloader):
        subbar = tqdm(total=7, desc=f"{i+1}/{args.num_batches}")
        batch = batch.to(device)
        labels = labels.to(device)

        # Infidelity
        subbar.set_postfix({"metric": "Infidelity"})
        batch_infidelity = infidelity(batch, labels, model, method,
                                      perturbation_range=list(np.linspace(.01, .3, 10)),
                                      num_perturbations=16)
        res["infidelity"].append(batch_infidelity)
        subbar.update(1)

        # Insertion curves
        subbar.set_postfix({"metric": "Insertion curves"})
        batch_ins_curves = insertion_deletion_curves(batch, labels, model, method, mask_range,
                                                     mask_value=0., mode="insertion")
        res["insertion"].append(batch_ins_curves)
        subbar.update(1)

        # Deletion curves
        subbar.set_postfix({"metric": "Deletion curves"})
        batch_del_curves = insertion_deletion_curves(batch, labels, model, method, mask_range,
                                                     mask_value=0., mode="deletion")
        res["deletion"].append(batch_del_curves)
        subbar.update(1)

        # Max-sensitivity
        subbar.set_postfix({"metric": "Max-sensitivity"})
        batch_max_sens = max_sensitivity(batch, labels, method,
                                         perturbation_range=list(np.linspace(.01, .3, 10)),
                                         num_perturbations=16)
        res["max-sens"].append(batch_max_sens)
        subbar.update(1)

        # Sensitivity-n
        subbar.set_postfix({"metric": "Sensitivity-n"})
        batch_sens_n = sensitivity_n(batch, labels, model, method, mask_range[1:],
                                     num_subsets=100, mask_value=0.)
        res["sens-n"].append(batch_sens_n)
        subbar.update(1)

        # Impact score (non-strict)
        subbar.set_postfix({"metric": "Impact Score"})
        batch_impact_score, b_i_count = impact_score(batch, labels, model, mask_range[1:], method,
                                                     mask_value=0., tau=.5, strict=False)
        res["impact"]["counts"].append(batch_impact_score)
        res["impact"]["total"] += b_i_count
        subbar.update(1)

        # Impact score (strict)
        subbar.set_postfix({"metric": "Strict Impact Score"})
        batch_s_impact_score, b_s_i_count = impact_score(batch, labels, model, mask_range[1:], method,
                                                         mask_value=0., strict=True)
        res["s-impact"]["counts"].append(batch_s_impact_score)
        res["s-impact"]["total"] += b_s_i_count

        subbar.update(1)
        subbar.set_postfix({})

    # Aggregate and save
    for name in ["infidelity", "insertion", "deletion", "sens-n"]:
        res[name] = torch.cat(res[name], dim=0)
        np.savetxt(path.join(args.out_dir, f"{name}.csv"), res[name].numpy(), delimiter=',')
    for name in ["impact", "s-impact"]:
        res[name]["counts"] = torch.sum(torch.stack(res[name]["counts"], dim=0), dim=0).numpy().tolist()
        with open(path.join(args.out_dir, f"{name}.json"), "w") as fp:
            json.dump(res[name], fp)
    res["max-sens"] = torch.max(torch.stack(res["max-sens"], dim=0), dim=0)[0]
    np.savetxt(path.join(args.out_dir, f"max-sens.csv"), res["max-sens"].numpy(), delimiter=",")

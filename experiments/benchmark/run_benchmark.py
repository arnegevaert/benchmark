import argparse
import itertools
from torch.utils.data import DataLoader
from attrbench.evaluation import *
from experiments.lib.util import get_ds_model, get_methods, get_mask_range
import numpy as np
import torch
from tqdm import tqdm
from os import path
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-n", "--num-batches", type=int)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--out-dir", type=str)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    # Parse arguments
    args = parser.parse_args()
    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # Get model, dataset, methods, params
    dataset, model = get_ds_model(args.dataset, args.model)
    methods = get_methods(model, args.batch_size, dataset.sample_shape[-2:], args.methods)
    model.to(device)
    model.eval()
    mask_range = get_mask_range(args.dataset)
    pert_range = list(np.linspace(.01, .3, 10))
    metadata = {
        "infidelity": {"x": pert_range},
        "insertion": {"x": mask_range},
        "deletion": {"x": mask_range},
        "max-sens": {"x": pert_range},
        "sens-n": {"x": mask_range[1:]},
        "impact": {"x": mask_range[1:]},
        "s-impact": {"x": mask_range[1:]}
    }

    # Validate metrics argument
    if args.metrics:
        for m in args.metrics:
            if m not in list(metadata.keys()):
                raise ValueError(f"Unrecognized metric: {m}")

    # Save metadata
    metrics = args.metrics if args.metrics else list(metadata.keys())
    with open(path.join(args.out_dir, "meta.json"), "w") as fp:
        json.dump({key: metadata[key] for key in metrics}, fp)

    # Run benchmark
    for i, m_name in enumerate(methods):
        method = methods[m_name]
        print(f"{m_name} ({i+1}/{len(methods)})")
        # Create subdir for method
        subdir = path.join(args.out_dir, m_name)
        if not path.isdir(subdir):
            os.makedirs(subdir)
        # Prepare results
        res = {}
        for name in metrics:
            if name in ["infidelity", "insertion", "deletion", "max-sens", "sens-n"]:
                res[name] = []
            elif name in ["impact", "s-impact"]:
                res[name] = {"counts": [], "total": 0}

        # Calculate metrics for each batch
        dataloader = itertools.islice(DataLoader(dataset, batch_size=args.batch_size), args.num_batches)
        for i, (batch, labels) in enumerate(dataloader):
            prog = tqdm(total=len(metrics), desc=f"{i + 1}/{args.num_batches}")
            batch = batch.to(device)
            labels = labels.to(device)

            # Infidelity
            if "infidelity" in metrics:
                prog.set_postfix({"metric": "Infidelity"})
                batch_infidelity = infidelity(batch, labels, model, method,
                                              perturbation_range=pert_range,
                                              num_perturbations=16)
                res["infidelity"].append(batch_infidelity)
                prog.update(1)

            # Insertion curves
            if "insertion" in metrics:
                prog.set_postfix({"metric": "Insertion curves"})
                batch_ins_curves = insertion_deletion_curves(batch, labels, model, method, mask_range,
                                                             mask_value=0., mode="insertion")
                res["insertion"].append(batch_ins_curves)
                prog.update(1)

            # Deletion curves
            if "deletion" in metrics:
                prog.set_postfix({"metric": "Deletion curves"})
                batch_del_curves = insertion_deletion_curves(batch, labels, model, method, mask_range,
                                                             mask_value=0., mode="deletion")
                res["deletion"].append(batch_del_curves)
                prog.update(1)

            # Max-sensitivity
            if "max-sens" in metrics:
                prog.set_postfix({"metric": "Max-sensitivity"})
                batch_max_sens = max_sensitivity(batch, labels, method,
                                                 perturbation_range=pert_range,
                                                 num_perturbations=16)
                res["max-sens"].append(batch_max_sens)
                prog.update(1)

            # Sensitivity-n
            if "sens-n" in metrics:
                prog.set_postfix({"metric": "Sensitivity-n"})
                batch_sens_n = sensitivity_n(batch, labels, model, method, mask_range[1:],
                                             num_subsets=16, mask_value=0.)
                res["sens-n"].append(batch_sens_n)
                prog.update(1)

            # Impact score (non-strict)
            if "impact" in metrics:
                prog.set_postfix({"metric": "Impact Score"})
                batch_impact_score, b_i_count = impact_score(batch, labels, model, mask_range[1:], method,
                                                             mask_value=0., tau=.5, strict=False)
                res["impact"]["counts"].append(batch_impact_score)
                res["impact"]["total"] += b_i_count
                prog.update(1)

            # Impact score (strict)
            if "s-impact" in metrics:
                prog.set_postfix({"metric": "Strict Impact Score"})
                batch_s_impact_score, b_s_i_count = impact_score(batch, labels, model, mask_range[1:], method,
                                                                 mask_value=0., strict=True)
                res["s-impact"]["counts"].append(batch_s_impact_score)
                res["s-impact"]["total"] += b_s_i_count
                prog.update(1)
            prog.close()

        # Aggregate and save
        for name in ["infidelity", "insertion", "deletion", "sens-n"]:
            if name in metrics:
                res[name] = torch.cat(res[name], dim=0)
                np.savetxt(path.join(subdir, f"{name}.csv"), res[name].numpy(), delimiter=',')
        for name in ["impact", "s-impact"]:
            if name in metrics:
                res[name]["counts"] = torch.sum(torch.stack(res[name]["counts"], dim=0), dim=0).numpy().tolist()
                with open(path.join(subdir, f"{name}.json"), "w") as fp:
                    json.dump(res[name], fp)
        if "max-sens" in metrics:
            res["max-sens"] = torch.cat(res["max-sens"], dim=0)
            np.savetxt(path.join(subdir, f"max-sens.csv"), res["max-sens"].numpy(), delimiter=",")

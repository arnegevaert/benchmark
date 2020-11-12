import argparse
import itertools
from torch.utils.data import DataLoader
from attrbench.evaluation.independent import *
from attrbench.lib import PixelMaskingPolicy
from experiments.lib.util import get_ds_model, get_methods, get_n_pixels
from experiments.lib.attribution import DimReplication
import numpy as np
import torch
from tqdm import tqdm
from os import path
import os
import json
import warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--out-dir", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--patch", type=str, default=None)
    parser.add_argument("--pixel-aggregation", type=str, choices=["avg", "max_abs"], default="avg")
    parser.add_argument("--num-workers", type=int, default=4)
    # Parse arguments
    args = parser.parse_args()
    if not path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # Get model, dataset, methods, params
    dataset, model = get_ds_model(args.dataset, args.model)
    methods = get_methods(model, args.pixel_aggregation, normalize=False, methods=args.methods,
                          batch_size=args.batch_size, sample_shape=dataset.sample_shape[-2:])
    masking_policy = PixelMaskingPolicy(mask_value=0.)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    #mask_range = get_mask_range(args.dataset)
    n_pixels = get_n_pixels(args.dataset)
    step_size = int(0.02 * n_pixels)
    mask_range = [step_size * i for i in range(25)]  # 0% to 50% of pixels in 25 steps
    pert_range = list(np.linspace(.01, 2., 20))
    metadata = {
        "infidelity": {"x": pert_range},
        "insertion": {"x": mask_range},
        "deletion": {"x": mask_range},
        "del-until-flip": {},
        "max-sens": {"x": pert_range},
        "sens-n": {"x": mask_range[1:]},
        "impact": {"x": mask_range[1:]},
        "s-impact": {"x": mask_range[1:]},
        "i-coverage": {}
    }

    # Validate metrics argument
    if args.metrics:
        for m in args.metrics:
            if m not in list(metadata.keys()):
                raise ValueError(f"Unrecognized metric: {m}")
    metrics = args.metrics if args.metrics else list(metadata.keys())

    # If no adversarial patch is given, we skip impact coverage
    if "i-coverage" in metrics and not args.patch:
        warnings.warn("No adversarial patch found. Skipping impact coverage.")
        metrics.remove("i-coverage")

    # Save metadata
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
            if name in ["impact", "s-impact"]:
                res[name] = {"counts": [], "total": 0}
            else:
                res[name] = []

        # Calculate metrics for each batch
        dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers))
        samples_done = 0
        while samples_done < args.num_samples:
            full_batch, full_labels = next(dataloader)
            full_batch = full_batch.to(device)
            full_labels = full_labels.to(device)

            # Only use correctly classified samples
            pred = torch.argmax(model(full_batch), dim=1)
            batch = full_batch[pred == full_labels]
            labels = full_labels[pred == full_labels]
            if batch.size(0) > 0:
                if samples_done + batch.size(0) > args.num_samples:
                    diff = args.num_samples - samples_done
                    batch = batch[:diff]
                    labels = labels[:diff]
                prog = tqdm(total=len(metrics), desc=f"{samples_done + 1}-{samples_done + batch.size(0)}/{args.num_samples}")

                # Infidelity
                if "infidelity" in metrics:
                    prog.set_postfix({"metric": "Infidelity"})
                    infid_method = DimReplication(method, 1, 3) if args.pixel_aggregation and batch.size(1) == 3 else method
                    batch_infidelity = infidelity(batch, labels, model, infid_method,
                                                perturbation_range=pert_range,
                                                num_perturbations=16)
                    res["infidelity"].append(batch_infidelity)
                    prog.update(1)

                # Insertion curves
                if "insertion" in metrics:
                    prog.set_postfix({"metric": "Insertion"})
                    batch_insertion = insertion_deletion(batch, labels, model, method, mask_range,
                                                        masking_policy, mode="insertion")
                    res["insertion"].append(batch_insertion)
                    prog.update(1)

                # Deletion curves
                if "deletion" in metrics:
                    prog.set_postfix({"metric": "Deletion"})
                    batch_deletion = insertion_deletion(batch, labels, model, method, mask_range,
                                                        masking_policy, mode="deletion")
                    res["deletion"].append(batch_deletion)
                    prog.update(1)

                # Deletion until flip
                if "del-until-flip" in metrics:
                    prog.set_postfix({"metric": "Deletion until flip"})
                    batch_del_flip = deletion_until_flip(batch, labels, model, method, masking_policy,
                                                        step_size=.01)
                    res["del-until-flip"].append(batch_del_flip)
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
                                                num_subsets=100, masking_policy=masking_policy)
                    res["sens-n"].append(batch_sens_n)
                    prog.update(1)

                # Impact score (non-strict)
                if "impact" in metrics:
                    prog.set_postfix({"metric": "Impact Score"})
                    batch_impact_score, b_i_count = impact_score(batch, labels, model, mask_range[1:], method,
                                                                masking_policy, tau=.5, strict=False)
                    res["impact"]["counts"].append(batch_impact_score)
                    res["impact"]["total"] += b_i_count
                    prog.update(1)

                # Impact score (strict)
                if "s-impact" in metrics:
                    prog.set_postfix({"metric": "Strict Impact Score"})
                    batch_s_impact_score, b_s_i_count = impact_score(batch, labels, model, mask_range[1:], method,
                                                                    masking_policy, strict=True)
                    res["s-impact"]["counts"].append(batch_s_impact_score)
                    res["s-impact"]["total"] += b_s_i_count
                    prog.update(1)

                # Impact coverage
                if "i-coverage" in metrics:
                    prog.set_postfix({"metric": "Impact Coverage"})
                    patch = torch.load(args.patch)
                    iou, keep = impact_coverage(batch, labels, model, method, patch, target_label=0)
                    res["i-coverage"].append(iou[keep])
                    prog.update(1)

                prog.close()
                samples_done += batch.size(0)

        # Aggregate and save
        for name in ["infidelity", "insertion", "deletion", "del-until-flip", "sens-n", "i-coverage", "max-sens"]:
            if name in metrics:
                res[name] = torch.cat(res[name], dim=0)
                np.savetxt(path.join(subdir, f"{name}.csv"), res[name].numpy(), delimiter=',')
        for name in ["impact", "s-impact"]:
            if name in metrics:
                res[name]["counts"] = torch.sum(torch.stack(res[name]["counts"], dim=0), dim=0).numpy().tolist()
                with open(path.join(subdir, f"{name}.json"), "w") as fp:
                    json.dump(res[name], fp)
import argparse
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
import warnings


if __name__ == "__main__":
    all_metrics = ["infidelity", "insertion", "deletion", "del-until-flip",
                   "max-sens", "sens-n", "impact", "s-impact", "i-coverage"]
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--out-dir", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+", default=None)
    parser.add_argument("--metrics", type=str, nargs="+", default=all_metrics, choices=all_metrics)
    parser.add_argument("--patch", type=str, default=None)
    parser.add_argument("--pixel-aggregation", type=str, choices=["avg", "max_abs"], default="avg")
    parser.add_argument("--num-workers", type=int, default=4)
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # Create/check output directory
    if not path.isdir(args.out_dir):
        print(f"Created directory {args.out_dir}")
        os.makedirs(args.out_dir)
    if len(os.listdir(args.out_dir)) > 0:
        warnings.warn(f"Directory {args.out_dir} is not empty")

    # Get model, dataset, methods, params
    dataset, model = get_ds_model(args.dataset, args.model)
    methods = get_methods(model, args.pixel_aggregation, normalize=False, methods=args.methods,
                          batch_size=args.batch_size, sample_shape=dataset.sample_shape[-2:])
    masking_policy = PixelMaskingPolicy(mask_value=0.)  # TODO configure this for different baselines
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    n_pixels = get_n_pixels(args.dataset)
    mask_range = list(np.rint(np.linspace(0, 1, 51) * n_pixels).astype(np.int))
    pert_range = list(np.linspace(.01, 2., 20))

    # If no adversarial patch is given, we skip impact coverage
    if "i-coverage" in args.metrics and not args.patch:
        warnings.warn("No adversarial patch found. Skipping impact coverage.")
        args.metrics.remove("i-coverage")

    dataloader = iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True))
    samples_done = 0
    res = {
        method: {
            metric: [] for metric in args.metrics
        } for method in methods
    }
    prog = tqdm(total=args.num_samples)
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

            batch_orig = batch.clone()

            for i, m_name in enumerate(methods):
                method = methods[m_name]
                # Infidelity
                if "infidelity" in args.metrics:
                    # Infidelity needs the attributions to have the same shape as the sample
                    infid_method = method
                    if args.pixel_aggregation and batch.size(1) == 3:
                        infid_method = DimReplication(method, dim=1, amount=3)
                    batch_infidelity = infidelity(batch, labels, model, infid_method,
                                                  perturbation_range=pert_range,
                                                  num_perturbations=16)
                    res[m_name]["infidelity"].append(batch_infidelity)

                # Insertion curves
                if "insertion" in args.metrics:
                    batch_insertion = insertion_deletion(batch, labels, model, method, mask_range,
                                                         masking_policy, mode="insertion")
                    res[m_name]["insertion"].append(batch_insertion)

                # Deletion curves
                if "deletion" in args.metrics:
                    batch_deletion = insertion_deletion(batch, labels, model, method, mask_range,
                                                        masking_policy, mode="deletion")
                    res[m_name]["deletion"].append(batch_deletion)

                # Deletion until flip
                if "del-until-flip" in args.metrics:
                    batch_del_flip = deletion_until_flip(batch, labels, model, method, masking_policy,
                                                         step_size=.01)
                    res[m_name]["del-until-flip"].append(batch_del_flip)

                # Max-sensitivity
                if "max-sens" in args.metrics:
                    batch_max_sens = max_sensitivity(batch, labels, method,
                                                     perturbation_range=pert_range,
                                                     num_perturbations=16)
                    res[m_name]["max-sens"].append(batch_max_sens)

                # Sensitivity-n
                if "sens-n" in args.metrics:
                    batch_sens_n = sensitivity_n(batch, labels, model, method, mask_range[1:-1],
                                                 num_subsets=16, masking_policy=masking_policy)
                    res[m_name]["sens-n"].append(batch_sens_n)

                # Impact score (non-strict)
                # Note: we can ignore the counts returned by (strict) impact score, since we already only use
                # correctly classified samples (i.e. the count is always equal to batch size)
                if "impact" in args.metrics:
                    batch_impact_score, _ = impact_score(batch, labels, model, mask_range[1:], method,
                                                         masking_policy, tau=.5, strict=False)
                    res[m_name]["impact"].append(batch_impact_score)

                # Impact score (strict)
                # Note: we can ignore the counts returned by (strict) impact score, since we already only use
                # correctly classified samples (i.e. the count is always equal to batch size)
                if "s-impact" in args.metrics:
                    batch_s_impact_score, _ = impact_score(batch, labels, model, mask_range[1:], method,
                                                           masking_policy, strict=True)
                    res[m_name]["s-impact"].append(batch_s_impact_score)

                # Impact coverage
                if "i-coverage" in args.metrics:
                    patch = torch.load(args.patch)
                    iou, keep = impact_coverage(batch, labels, model, method, patch, target_label=0)
                    res[m_name]["i-coverage"].append(iou[keep])
            samples_done += batch.size(0)
            prog.update(batch.size(0))
            print(torch.sum(batch_orig - batch))

    # Aggregate and save
    headers = {
        "infidelity": ','.join(map(str, pert_range)),
        "max-sens": ','.join(map(str, pert_range)),
        "insertion": ','.join(map(str, mask_range)),
        "deletion": ','.join(map(str, mask_range)),
        "sens-n": ','.join(map(str, mask_range[1:-1])),
        "impact": ','.join(map(str, mask_range[1:])),
        "s-impact": ','.join(map(str, mask_range[1:])),
        "del-until-flip": '',
        "i-coverage": ''
    }

    for m_name in res:
        # Create subdir for method
        subdir = path.join(args.out_dir, m_name)
        if not path.isdir(subdir):
            os.makedirs(subdir)
        for metric in res[m_name]:
            if metric in ["impact", "s-impact"]:
                # Sum the counts and divide by total number of samples
                np.savetxt(path.join(subdir, f"{metric}.csv"),
                           (torch.sum(torch.stack(res[m_name][metric], dim=0), dim=0).float() / args.num_samples).numpy(),
                           header=headers[metric],
                           delimiter=',')
            else:
                np.savetxt(path.join(subdir, f"{metric}.csv"),
                           torch.cat(res[m_name][metric], dim=0).numpy(),
                           header=headers[metric],
                           delimiter=',')

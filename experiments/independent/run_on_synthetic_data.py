import argparse
import numpy as np
from os import path
from torch import nn
import torch
import json
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from experiments.lib.util import get_methods
from attrbench.functional import *
from attrbench.lib import FeatureMaskingPolicy
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--num-samples", type=int, default=1024)
    args = parser.parse_args()
    device = "cpu"
    batch_size = 64

    # Load data and model
    data = {
        key: np.load(path.join(args.data_dir, f"{key}.npy"))
        for key in ("x_train", "x_test", "y_train", "y_test")
    }
    mask_range = list(np.rint(np.linspace(0, 1, 51) * data["x_train"].shape[1]).astype(np.int))
    mask_range = sorted(list(set(mask_range)))
    pert_range = list(np.linspace(.01, 2., 20))
    ds = TensorDataset(torch.tensor(data["x_test"], dtype=torch.float32), torch.tensor(data["y_test"]))
    dl = iter(DataLoader(ds, batch_size=batch_size, num_workers=8))
    with open(path.join(args.data_dir, "args.json")) as fp:
        data_args = json.load(fp)
    model = nn.Sequential(
        nn.Linear(data_args["num_features"], data_args["num_nodes"]),
        nn.ReLU(),
        nn.Linear(data_args["num_nodes"], data_args["num_classes"])
    )
    model.load_state_dict(torch.load(path.join(args.data_dir, "model.pt")))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    method_names = ["Gradient", "IntegratedGradients", "SmoothGrad", "InputXGradient", "GuidedBackprop", "Random"]
    metrics = ["infidelity", "insertion", "deletion", "del-until-flip", "max-sens", "sens-n", "impact", "s-impact"]
    methods = get_methods(model, aggregation_fn=None, normalize=False,
                          batch_size=batch_size, methods=method_names, sample_shape=data["x_train"].shape[1])
    fmp = FeatureMaskingPolicy(mask_value=0.)

    res = {
        method: {
            metric: [] for metric in metrics
        } for method in methods
    }
    samples_done = 0
    prog = tqdm(total=args.num_samples)
    while samples_done < args.num_samples:
        all_samples, all_labels = next(dl)
        all_samples = all_samples.to(device)
        all_labels = all_labels.to(device)
        pred = torch.argmax(model(all_samples), dim=1)
        samples = all_samples[pred == all_labels]
        labels = all_labels[pred == all_labels]
        if samples.size(0) > 0:
            if samples_done + samples.size(0) > args.num_samples:
                diff = args.num_samples - samples_done
                samples = samples[:diff]
                labels = labels[:diff]

            for m_name in methods:
                method = methods[m_name]
                res[m_name]["infidelity"].append(
                    infidelity(samples, labels, model, method, perturbation_range=pert_range, num_perturbations=16)
                )
                res[m_name]["insertion"].append(
                    insertion_deletion(samples, labels, model, method, mask_range, fmp, mode="insertion")
                )
                res[m_name]["deletion"].append(
                    insertion_deletion(samples, labels, model, method, mask_range, fmp, mode="deletion")
                )
                res[m_name]["del-until-flip"].append(
                    deletion_until_flip(samples, labels, model, method, step_size=.01, masking_policy=fmp)
                )
                res[m_name]["max-sens"].append(
                    max_sensitivity(samples, labels, method, perturbation_range=pert_range, num_perturbations=16)
                )
                res[m_name]["sens-n"].append(
                    sensitivity_n(samples, labels, model, method, mask_range[1:-1], num_subsets=16, masking_policy=fmp)
                )
                res[m_name]["impact"].append(
                    impact_score(samples, labels, model, method, mask_range[1:], strict=False, masking_policy=fmp,
                                 tau=.5)[0]
                )
                res[m_name]["s-impact"].append(
                    impact_score(samples, labels, model, method, mask_range[1:], strict=True, masking_policy=fmp)[0]
                )
        samples_done += samples.size(0)
        prog.update(samples.size(0))

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

import argparse
import torch
import numpy as np
import logging
from experiments.general_imaging.dataset_models import get_dataset_model
from torch.utils.data import DataLoader
from experiments.lib import MethodLoader
from os import path
import os
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_config", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--multi-label", action="store_true")
    parser.add_argument("--explain-label", type=int, default=None)

    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    if not path.isdir(args.output):
        os.makedirs(args.output)

    ds, model, _ = get_dataset_model(args.dataset, model_name=args.model)
    model.to(device)
    model.eval()
    ml = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                      reference_dataset=ds)
    methods = ml.load_config(args.method_config)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    it = iter(dl)
    done = 0
    all_samples = []
    attrs = {method: [] for method in methods}
    prog = tqdm(total=args.num_samples)
    while done < args.num_samples:
        full_batch, full_labels = next(it)
        full_batch = full_batch.to(device)
        full_labels = full_labels.to(device)

        with torch.no_grad():
            out = model(full_batch)
            pred = torch.argmax(out, dim=1)
            if args.multi_label:
                pred_labels = full_labels[torch.arange(len(pred)), pred]
                batch = full_batch[pred_labels == 1]
                labels = pred[pred_labels == 1]
            else:
                batch = full_batch[pred == full_labels]
                labels = full_labels[pred == full_labels]
                if args.explain_label is not None:
                    batch = batch[labels == args.explain_label]
                    labels = labels[labels == args.explain_label]
            if done + batch.size(0) > args.num_samples:
                diff = args.num_samples - done
                batch = batch[:diff]
                labels = labels[:diff]
        all_samples.append(batch.cpu().detach().numpy())
        for i, method in enumerate(methods.keys()):
            prog.set_postfix({"method": f"{method} ({i+1}/{len(methods.keys())}"})
            attrs[method].append(methods[method](batch, labels).cpu().detach().numpy())
        done += batch.shape[0]
        prog.update(batch.shape[0])

    np.save(path.join(args.output, "samples.npy"), np.concatenate(all_samples, axis=0))
    for method in methods:
        np.save(path.join(args.output, f"{method}.npy"), np.concatenate(attrs[method], axis=0))

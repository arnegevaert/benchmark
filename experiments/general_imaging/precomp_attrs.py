import argparse
import logging
import multiprocessing
import os
from os import path

import numpy as np
import torch

from attrbench.suite import PrecomputedAttrsSuite
from experiments.general_imaging.dataset_models import get_dataset_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_config", type=str)
    parser.add_argument("attrs_dir", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-t", "--num_workers", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--multi_label", action="store_true")
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    # Get dataset, model, methods
    _, model, _ = get_dataset_model(args.dataset, model_name=args.model)

    # Load attrs from disk
    attrs = {}
    samples = np.load(path.join(args.attrs_dir, "samples.npy"))
    for filename in os.listdir(args.attrs_dir):
        if filename != "samples.npy":
            attrs[filename.split(".")[0]] = np.load(path.join(args.attrs_dir, filename))

    # Run BM suite and save result to disk
    bm_suite = PrecomputedAttrsSuite(model, attrs, samples, args.batch_size, device, seed=args.seed,
                                     log_dir=args.log_dir, explain_label=args.explain_label,
                                     multi_label=args.multi_label, num_workers=args.num_workers)
    bm_suite.load_config(args.suite_config)

    bm_suite.run(verbose=True)

    bm_suite.save_result(args.output)

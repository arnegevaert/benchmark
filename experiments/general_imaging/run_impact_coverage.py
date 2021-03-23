import argparse
import torch
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from experiments.lib import MethodLoader
from attrbench.suite import Suite
from attrbench.metrics import ImpactCoverage
from torch.utils.data import DataLoader
import logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_config", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--multi_label", action="store_true")
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    ds, model, patch_folder = get_dataset_model(args.dataset, model_name=args.model)
    ml = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                      reference_dataset=ds)
    methods = ml.load_config(args.method_config)

    # Run BM suite and save result to disk
    bm_suite = Suite(model, methods,
                     DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4),
                     device,
                     seed=args.seed,
                     patch_folder=patch_folder,
                     multi_label=args.multi_label,
                     log_dir=args.log_dir)

    ic = ImpactCoverage(model, methods, patch_folder, args.log_dir)
    bm_suite.metrics = {"impact_coverage": ic}

    bm_suite.run(args.num_samples, verbose=True)
    bm_suite.save_result(args.output)

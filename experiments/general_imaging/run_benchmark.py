import argparse
import torch
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from experiments.lib import MethodLoader
from attrbench.suite import Suite, MetricLoader
from torch.utils.data import DataLoader
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_config", type=str)
    parser.add_argument("method_config", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-i", "--save-images", action="store_true")
    parser.add_argument("-a", "--save-attrs", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--multi_label", action="store_true")
    parser.add_argument("--explain_label", type=int, default=None)
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    logging.basicConfig(
        format='[%(asctime)s %(levelname)s] %(message)s',
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info("Saving images" if args.save_images else "Not saving images")
    logging.info("Saving attributions" if args.save_images else "Not saving attributions")

    # Get dataset, model, methods
    ds, model, patch_folder = get_dataset_model(args.dataset, model_name=args.model)
    methods = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                           reference_dataset=ds).load_config(args.method_config)

    # Get metrics
    metrics = MetricLoader(args.suite_config, model, methods, args.log_dir, patch_folder=patch_folder).load()

    # Run BM suite and save result to disk
    bm_suite = Suite(model, methods, metrics, device, log_dir=args.log_dir, multi_label=args.multi_label, explain_label=args.explain_label)
    bm_suite.run(DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4), args.num_samples,
                 args.seed, args.save_images, args.save_attrs, args.output)

import argparse
import torch
from experiments.general_imaging.dataset_models import get_dataset_model
from experiments.lib import MethodLoader
from attrbench.suite import Suite
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_config", type=str)
    parser.add_argument("method_config", nargs="?", type=str, default="config/methods/default.yaml")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-i", "--save-images", action="store_true")
    parser.add_argument("-a", "--save-attrs", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    # Parse arguments
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    print("Saving images" if args.save_images else "Not saving images")
    print("Saving attributions" if args.save_attrs else "Not saving attributions")

    # Get dataset, model, methods
    ds, model, sample_shape, patch_folder = get_dataset_model(args.dataset)
    ml = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                      sample_shape=sample_shape, reference_dataset=ds)
    methods = ml.load_config("config/methods/default.yaml")

    # Run BM suite and save result to disk
    bm_suite = Suite(model, methods,
                     DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4),
                     device,
                     save_images=args.save_images,
                     save_attrs=args.save_attrs,
                     seed=args.seed,
                     patch_folder=patch_folder)
    bm_suite.load_config(args.suite_config)
    bm_suite.run(args.num_samples, verbose=True)
    bm_suite.save_result(args.output)

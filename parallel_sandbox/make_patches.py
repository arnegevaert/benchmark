from util.get_dataset_model import get_dataset, get_model
from attrbench.distributed.metrics import MakePatches
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-patches", type=int, default=10)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-o", "--output-dir", type=str, default="patches")
    args = parser.parse_args()

    make_patches = MakePatches(get_model, get_dataset(), num_patches=args.num_patches,
                               batch_size=args.batch_size, target_label=0, path=args.output_dir)
    make_patches.run()

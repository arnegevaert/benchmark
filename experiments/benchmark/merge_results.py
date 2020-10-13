import os
import argparse
from os import path


if __name__ == "__main__":
    """
        This script takes all benchmark result files from a source directory and puts them
        in their respective place in the destination directory. Use this to merge
        results if a subset of metrics was (re-)run (instead of manually moving csv or json
        files around).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Source directory")
    parser.add_argument("--dst", type=str, help="Destination directory")
    parser.add_argument("--dry", action="store_true", help="Dry-run: just list changes without executing")
    args = parser.parse_args()

    datasets = os.listdir(args.src)
    for ds in datasets:
        ds_dir = path.join(args.src, ds)
        methods = [m for m in os.listdir(ds_dir) if path.isdir(path.join(ds_dir, m))]
        for m in methods:
            filenames = os.listdir(path.join(ds_dir, m))
            for f in filenames:
                if not args.dry:
                    os.rename(path.join(ds_dir, m, f), path.join(args.dst, ds, m, f))
                    print(f"Moved {path.join(ds_dir, m, f)} to {path.join(args.dst, ds, m, f)}")
                else:
                    print(f"{path.join(ds_dir, m, f)} -> {path.join(args.dst, ds, m, f)}")

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

    methods = [m for m in os.listdir(args.src) if path.isdir(path.join(args.src, m))]
    for m in methods:
        filenames = os.listdir(path.join(args.src, m))
        for f in filenames:
            if not args.dry:
                os.rename(path.join(args.src, m, f), path.join(args.dst, m, f))
                print(f"Moved {path.join(args.src, m, f)} to {path.join(args.dst, m, f)}")
            else:
                print(f"{path.join(args.src, m, f)} -> {path.join(args.dst, m, f)}")

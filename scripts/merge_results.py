"""
Merges two result files
Requires both files to have the same selection of methods
Assumes both files were run on the same samples
All metrics from src will be added to dst
If metrics in src were already present in dst, they will be overwritten
"""
import argparse
from attrbench.suite import Result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str)
    parser.add_argument("dst", type=str)
    parser.add_argument("outfile", type=str)
    args = parser.parse_args()

    src = Result.load_hdf(args.src)
    dst = Result.load_hdf(args.dst)
    for metric_name in src.get_metrics():
        dst.data[metric_name] = src.data[metric_name]
    dst.save_hdf(args.outfile)

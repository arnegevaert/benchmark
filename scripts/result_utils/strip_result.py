"""
Removes images and attributions from result file
"""
import argparse
from attrbench.suite import SuiteResult


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    args = parser.parse_args()

    hdf_obj = SuiteResult.load_hdf(args.infile)
    hdf_obj.attributions = None
    hdf_obj.images = None
    hdf_obj.save_hdf(args.outfile)
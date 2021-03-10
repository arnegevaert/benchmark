"""
This script can be used to check if loading and saving the SuiteResult works correctly,
by loading and saving the same file and then checking if the result is identical (using h5diff)
"""
from attrbench.suite import SuiteResult
import sys


if __name__ == "__main__":
    res = SuiteResult.load_hdf(sys.argv[1])
    res.save_hdf(sys.argv[2])

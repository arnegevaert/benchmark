import argparse
from attrbench.suite import SuiteResult, Dashboard
import attrbench.suite.dashboard.util as util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-p", "--port", type=int, default=9000)
    parser.add_argument("--metrics", type=str, nargs="*", default=None)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    args = parser.parse_args()

    sres = SuiteResult.load_hdf(args.file)

    dfs = util.get_dfs(sres,mode='raw',masker='constant',activation='linear')
    db = Dashboard(sres, args.file, port=args.port)
    db.run()

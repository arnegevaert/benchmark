import argparse
from attrbench.suite import Result, Dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-p", "--port", type=int, default=9000)
    parser.add_argument("--metrics", type=str, nargs="*", default=None)
    parser.add_argument("--methods", type=str, nargs="*", default=None)
    args = parser.parse_args()

    hdf_obj = Result.load_hdf(args.file, metrics=args.metrics, methods=args.methods)
    db = Dashboard(hdf_obj, args.file, port=args.port)
    db.run()

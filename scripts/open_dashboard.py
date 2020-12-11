import argparse
import webbrowser
from attrbench.suite import Result
from attrbench.suite import Dashboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("-p", "--port", type=int, default=9000)
    args = parser.parse_args()

    hdf_obj = Result.load_hdf(args.file)
    db = Dashboard(hdf_obj, args.file, port=args.port)
    db.run()
    webbrowser.open_new(f"localhost:{args.port}")

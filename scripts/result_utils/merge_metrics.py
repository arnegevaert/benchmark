import argparse
import sys
import h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge metric results from source_file into dest_file,"
                                                 " and write output to out_file."
                                                 " Assumes the same methods are present in both files.")
    parser.add_argument("source_file", type=str)
    parser.add_argument("dest_file", type=str)
    parser.add_argument("out_file", type=str)
    args = parser.parse_args()

    with h5py.File(args.source_file, mode="r") as source_file, \
            h5py.File(args.dest_file, mode="r") as dest_file, \
            h5py.File(args.out_file, mode="w") as out_file:
        # Copy dest file to out file
        for key in dest_file.keys():
            dest_file.copy(key, out_file)
        out_file.attrs["num_samples"] = dest_file.attrs["num_samples"]

        # Copy each metric from source file to out file
        source_res = source_file["results"]
        out_res = out_file["results"]
        for key in source_res.keys():
            if key not in out_res.keys():
                source_res.copy(key, out_res)
            else:
                sys.exit(f"Metric {key} already present in destination file, skipping")

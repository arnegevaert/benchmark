import argparse
import h5py
import sys


def merge_rec(source_node, dest_node, out_node, allow_overwrite):
    # Check if keys are identical
    keys = sorted(list(source_node.keys()))

    for key in keys:
        if type(source_node[key]) == h5py.Group:
            if type(source_node[key]) != type(dest_node[key]):
                sys.exit(f"Key types don't match: {source_node.name}, {dest_node.name}")
            # Recursively descend
            merge_rec(source_node[key], dest_node[key], out_node[key], allow_overwrite)
        elif type(source_node[key]) == h5py.Dataset:
            # Copy to out_node
            if key not in out_node.keys() or allow_overwrite:
                out_node.create_dataset(key, data=source_node[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge method results from source_file into dest_file,"
                                                 " and write output to out_file."
                                                 " Assumes the same metrics are present in both files."
                                                 " Methods present in both files will NOT be overwritten by default."
                                                 " Pass --allow_overwrite to override this setting.")
    parser.add_argument("source_file", type=str)
    parser.add_argument("dest_file", type=str)
    parser.add_argument("out_file", type=str)
    parser.add_argument("--allow_overwrite", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.source_file, mode="r") as source_file, \
            h5py.File(args.dest_file, mode="r") as dest_file, \
            h5py.File(args.out_file, mode="w") as out_file:
        # Check if metric names are identical
        metric_names = sorted(list(source_file["results"].keys()))
        if not sorted(list(dest_file["results"].keys())) == metric_names:
            sys.exit("Metric names must be identical in order to merge.")

        # Copy dest file to out file
        for key in dest_file.keys():
            dest_file.copy(key, out_file)
        out_file.attrs["num_samples"] = dest_file.attrs["num_samples"]

        merge_rec(source_file["results"], dest_file["results"], out_file["results"], args.allow_overwrite)

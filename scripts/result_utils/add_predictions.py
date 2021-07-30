import h5py
import numpy as np
from os import path


if __name__ == "__main__":
    src_dir = "../../out"
    out_dir = "../../out/confidence_merged"

    for ds in ["mnist", "fashionmnist", "cifar10", "cifar100", "svhn", "imagenet", "caltech", "places"]:
        with h5py.File(path.join(src_dir, f"{ds}.h5"), mode="r") as source_file, \
             h5py.File(path.join(out_dir, f"{ds}.h5"), mode="w") as out_file:
            # Copy source to out
            for key in source_file.keys():
                source_file.copy(key, out_file)
            out_file.attrs["num_samples"] = source_file.attrs["num_samples"]

            # Read CSV file with predictions
            preds = np.loadtxt(path.join(src_dir, "confidence", f"{ds}.csv"), delimiter=",")
            out_file.create_dataset("predictions", data=preds)

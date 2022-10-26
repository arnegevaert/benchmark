from util.get_dataset_model import get_model
from attrbench.distributed.metrics.deletion import DistributedDeletion
from attrbench.data import AttributionsDataset, HDF5Dataset
from attrbench.lib.masking import ConstantMasker
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples-dataset", type=str, default="samples.h5")
    parser.add_argument("-a", "--attrs-dataset", type=str, default="attributions.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-o", "--output-file", type=str, default="deletion.h5")
    args = parser.parse_args()

    dataset = AttributionsDataset(HDF5Dataset(args.samples_dataset), args.attrs_dataset,
                                  aggregate_fn=lambda a: np.mean(a, axis=0))
    deletion = DistributedDeletion(get_model, dataset, args.batch_size,
                                   maskers={"constant": ConstantMasker(feature_level="channel")},
                                   activation_fns="linear")
    deletion.run()
    print("Done")

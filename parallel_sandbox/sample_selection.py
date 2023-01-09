from util.get_dataset_model import get_dataset, get_model
from attrbench.data import HDF5DatasetWriter
from attrbench.distributed import SampleSelection
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", type=int, default=32)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-o", "--output-file", type=str, default="samples.h5")
    args = parser.parse_args()

    writer = HDF5DatasetWriter(path="samples.h5", num_samples=args.num_samples, sample_shape=(3, 224, 224))
    sample_selection = SampleSelection(get_model, get_dataset(), writer, num_samples=args.num_samples, batch_size=args.batch_size)
    sample_selection.run()

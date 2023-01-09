from util.get_dataset_model import get_model
from attrbench.metrics import Irof
from attrbench.data import AttributionsDataset, HDF5Dataset
from attrbench.masking import ConstantMasker
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples-dataset", type=str, default="samples.h5")
    parser.add_argument("-a", "--attrs-dataset", type=str, default="attributions.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-o", "--output-file", type=str, default="irof.h5")
    args = parser.parse_args()

    dataset = AttributionsDataset(HDF5Dataset(args.samples_dataset), args.attrs_dataset,
                                  aggregate_axis=0, aggregate_method="mean")
    deletion = Irof(get_model, dataset, args.batch_size,
                    maskers={"constant": ConstantMasker(feature_level="pixel")},
                    activation_fns="linear")
    deletion.run(result_path=args.output_file)

from attrbench.util.method_factory import MethodFactory
from util.get_dataset_model import get_model
from attrbench.distributed import AttributionsComputation
from attrbench.data import HDF5Dataset, AttributionsDatasetWriter
from methods import Gradient, InputXGradient, IntegratedGradients, Random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="samples.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-o", "--output-file", type=str, default="attributions.h5")
    args = parser.parse_args()

    dataset = HDF5Dataset(args.dataset)
    writer = AttributionsDatasetWriter(args.output_file, truncate=True, num_samples=len(dataset),
                                       sample_shape=dataset.sample_shape)

    method_factory = MethodFactory({
        "Gradient": Gradient,
        "InputXGradient": InputXGradient,
        "IntegratedGradients": (IntegratedGradients, {"batch_size": args.batch_size}),
        "Random": Random
        })

    computation = AttributionsComputation(get_model, method_factory, dataset,
                                          batch_size=args.batch_size,
                                          writer=writer)
    computation.run()

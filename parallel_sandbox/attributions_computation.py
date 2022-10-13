from util.get_dataset_model import get_model
from attrbench.distributed import AttributionsComputation
from attrbench.data import HDF5Dataset, AttributionsDatasetWriter
from captum import attr
import argparse


def method_factory(model):
    saliency = attr.Saliency(model)
    ixg = attr.InputXGradient(model)
    ig = attr.IntegratedGradients(model)
    return {
        "Gradient": saliency.attribute,
        "InputXGradient": ixg.attribute,
        "IntegratedGradients": lambda x, y: ig.attribute(inputs=x, target=y, internal_batch_size=args.batch_size)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="samples.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-o", "--output-file", type=str, default="attributions.h5")
    args = parser.parse_args()

    dataset = HDF5Dataset(args.dataset)
    writer = AttributionsDatasetWriter(args.output_file, truncate=True, num_samples=len(dataset),
                                       sample_shape=dataset.sample_shape)
    computation = AttributionsComputation(get_model, method_factory, dataset, batch_size=args.batch_size, writer=writer)
    computation.start()

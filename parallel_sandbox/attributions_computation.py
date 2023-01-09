from util.get_dataset_model import get_model
from attrbench.distributed import AttributionsComputation
from attrbench.data import HDF5Dataset, AttributionsDatasetWriter
from captum import attr
import torch
import argparse


class MethodFactory:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, model):
        saliency = attr.Saliency(model)
        ixg = attr.InputXGradient(model)
        ig = attr.IntegratedGradients(model)
        # TODO define a "AttributionMethod" wrapper class that contains a function and an is_baseline property
        return {
            "Gradient": saliency.attribute,
            "InputXGradient": ixg.attribute,
            "IntegratedGradients": lambda x, y: ig.attribute(inputs=x, target=y, internal_batch_size=self.batch_size),
            "Random": lambda x, _: torch.rand_like(x)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="samples.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-o", "--output-file", type=str, default="attributions.h5")
    args = parser.parse_args()

    dataset = HDF5Dataset(args.dataset)
    writer = AttributionsDatasetWriter(args.output_file, truncate=True, num_samples=len(dataset),
                                       sample_shape=dataset.sample_shape)
    computation = AttributionsComputation(get_model, MethodFactory(args.batch_size), dataset, batch_size=args.batch_size, writer=writer)
    computation.run()

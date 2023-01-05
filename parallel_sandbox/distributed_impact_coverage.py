from attrbench.util import MethodFactory
from util.get_dataset_model import get_model
from attrbench.distributed.metrics import DistributedImpactCoverage
from attrbench.data import HDF5Dataset, IndexDataset
import argparse
from captum import attr
import torch

class SimpleMethodFactory(MethodFactory):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._method_names = ["Gradient", "InputXGradient",
                              "IntegratedGradients", "Random"]

    def get_method_names(self):
        return self._method_names

    def __call__(self, model):
        saliency = attr.Saliency(model)
        ixg = attr.InputXGradient(model)
        ig = attr.IntegratedGradients(model)
        # TODO define a "AttributionMethod" wrapper class that contains a
        # function and an is_baseline property
        return {
            "Gradient": saliency.attribute,
            "InputXGradient": ixg.attribute,
            "IntegratedGradients": lambda x, target: ig.attribute(
                inputs=x, target=target, internal_batch_size=self.batch_size),
            "Random": lambda x, target: torch.rand_like(x)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="samples.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-o", "--output-file", type=str, default="coverage.h5")
    parser.add_argument("-p", "--patch-folder", type=str, default="patches")
    args = parser.parse_args()

    dataset = IndexDataset(HDF5Dataset(args.dataset))
    coverage = DistributedImpactCoverage(get_model, dataset, args.batch_size,
                                         SimpleMethodFactory(args.batch_size),
                                         args.patch_folder)
    coverage.run(result_path=args.output_file)

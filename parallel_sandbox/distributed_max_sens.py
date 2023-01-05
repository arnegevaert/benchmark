from attrbench.data.attributions_dataset import AttributionsDataset
from attrbench.util import MethodFactory
from util.get_dataset_model import get_model
from attrbench.distributed.metrics import DistributedMaxSensitivity
from attrbench.data import HDF5Dataset, IndexDataset
import argparse
from captum import attr
import torch

class SimpleMethodFactory(MethodFactory):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self._method_names = ["Gradient", "InputXGradient", "Random"]

    def get_method_names(self):
        return self._method_names

    def __call__(self, model):
        saliency = attr.Saliency(model)
        ixg = attr.InputXGradient(model)
        # TODO define a "AttributionMethod" wrapper class that contains a
        # function and an is_baseline property
        return {
            "Gradient": saliency.attribute,
            "InputXGradient": ixg.attribute,
            "Random": lambda x, target: torch.rand_like(x)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples-dataset", type=str, default="samples.h5")
    parser.add_argument("-a", "--attrs-dataset", type=str, default="attributions.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-o", "--output-file", type=str, default="maxsens.h5")
    args = parser.parse_args()

    dataset = AttributionsDataset(HDF5Dataset(args.samples_dataset),
                                  args.attrs_dataset,
                                  group_attributions=True)

    maxsens = DistributedMaxSensitivity(get_model, dataset, args.batch_size,
                                        SimpleMethodFactory(args.batch_size),
                                        num_perturbations=50, radius=0.01)
    maxsens.run(result_path=args.output_file)

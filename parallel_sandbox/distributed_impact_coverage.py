from attrbench.util import MethodFactory
from attrbench.util.model import Model
from methods import Gradient, InputXGradient, IntegratedGradients, Random
from util.get_dataset_model import get_model
from attrbench.metrics import ImpactCoverage
from attrbench.data import HDF5Dataset, IndexDataset
import argparse
from captum import attr
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="samples.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-o", "--output-file", type=str, default="coverage.h5")
    parser.add_argument("-p", "--patch-folder", type=str, default="patches")
    args = parser.parse_args()
    
    method_factory = MethodFactory({
        "Gradient": Gradient,
        "InputXGradient": InputXGradient,
        "IntegratedGradients": (IntegratedGradients, {"batch_size": args.batch_size}),
        "Random": Random
        })

    dataset = IndexDataset(HDF5Dataset(args.dataset))
    resnet = get_model()
    coverage = ImpactCoverage(Model(resnet), dataset, args.batch_size,
                              method_factory,
                              args.patch_folder)
    coverage.run(result_path=args.output_file)

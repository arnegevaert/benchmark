from attrbench.data.attributions_dataset import AttributionsDataset
from attrbench.util import MethodFactory
from attrbench.util.model import Model
from methods import Gradient, InputXGradient, IntegratedGradients, Random
from util.get_dataset_model import get_model
from attrbench.metrics import MaxSensitivity
from attrbench.data import HDF5Dataset
import argparse


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

    resnet = get_model()
    
    method_factory = MethodFactory({
        "Gradient": Gradient,
        "InputXGradient": InputXGradient,
        "IntegratedGradients": (IntegratedGradients, {"batch_size": args.batch_size}),
        "Random": Random
        })

    maxsens = MaxSensitivity(Model(resnet), dataset, args.batch_size,
                             method_factory,
                             num_perturbations=50, radius=0.01)
    maxsens.run(result_path=args.output_file)

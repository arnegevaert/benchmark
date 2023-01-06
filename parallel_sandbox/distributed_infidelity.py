from util.get_dataset_model import get_model
from attrbench.metrics import Infidelity
from attrbench.data import AttributionsDataset, HDF5Dataset
from attrbench.metrics.infidelity import GaussianPerturbationGenerator
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--samples-dataset", type=str, default="samples.h5")
    parser.add_argument("-a", "--attrs-dataset", type=str, default="attributions.h5")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-o", "--output-file", type=str, default="sensn.h5")
    args = parser.parse_args()

    dataset = AttributionsDataset(HDF5Dataset(args.samples_dataset), args.attrs_dataset,
                                  aggregate_axis=0, aggregate_method="mean", group_attributions=True)
    pert_gens = {
        "gaussian": GaussianPerturbationGenerator(sd=0.1),
        #"square": SquarePerturbationGenerator(square_size=0.1),
        #"noisy_bl": NoisyBaselinePerturbationGenerator(sd=0.1),
        #"segment": SegmentRemovalPerturbationGenerator(num_segments=100)
    }
    infidelity = Infidelity(get_model, dataset, args.batch_size, pert_gens, num_perturbations=1000,
                            activation_fns=("linear",))
    infidelity.run(result_path=args.output_file)

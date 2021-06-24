import argparse
import torch
from experiments.general_imaging.lib.dataset_models import get_dataset_model
from attrbench.metrics import Runtime
from experiments.lib import MethodLoader
from attrbench.suite import Suite
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_config", type=str)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    parser.add_argument("-c", "--cuda", action="store_true")
    parser.add_argument("-o", "--output", type=str, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    ds, model, _ = get_dataset_model(args.dataset, model_name=args.model)
    methods = MethodLoader(model=model, last_conv_layer=model.get_last_conv_layer(),
                           reference_dataset=ds).load_config(args.method_config)

    metrics = {"runtime": Runtime(model, methods),
               "runtime_single": Runtime(model, methods, single_image=True)}

    suite = Suite(model, methods, metrics, device)
    suite.run(DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4), args.num_samples,
              out_filename=args.output)

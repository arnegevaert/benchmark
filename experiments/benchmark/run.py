import argparse
from attrbench.evaluation import impact_score, impact_coverage, infidelity,\
    insertion_deletion_curves, max_sensitivity, sensitivity_n
from experiments.benchmark.util import get_ds_model_method


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    dataset, model, method = get_ds_model_method(args.dataset, args.model, args.method, args.batch_size)

    for i in range(args.num_batches):
        print(f"Batch {i+1}/{args.num_batches}...")

        # Infidelity
        print("Infidelity...")

        # Insertion curves
        print("Insertion curves...")

        # Deletion curves
        print("Deletion curves...")

        # Max-sensitivity
        print("Max-sensitivity...")

        # Sensitivity-n
        print("Sensitivity-n...")

        # Impact score (strict)
        print("Strict impact score...")

        # Impact score (non-strict)
        print("Non-strict impact score...")

        # Impact coverage
        print("Impact coverage...")

        print(f"Batch {i+1}/{args.num_batches} finished.")
        print()

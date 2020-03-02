import argparse
from arguments import Arguments

# Constants
METHODS = [
    "DeepLift",
    "Gradient",
    "InputXGradient",
    "IntegratedGradients"
]

# Specify arguments
parser = argparse.ArgumentParser(description="A benchmark for XAI techniques.")
parser.add_argument("--cuda", action="store_true", help="Use CUDA device if available")
parser.add_argument("--metric", action="store", choices=["relevance", "robustness"],
                    default="relevance")
parser.add_argument("--perturbation", action="store", choices=["mean-shift", "noise"])

# Parse arguments
args = parser.parse_args()

# Manual checking of arguments
if args.metric is "robustness" and not args.perturbation:
    parser.error("Robustness metric needs a type of perturbation (choose from mean-shift, noise)")

# Create Arguments object
Arguments(vars(args))



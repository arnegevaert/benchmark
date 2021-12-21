# AttrBench: Metrics for Feature Attribution Techniques
This repository contains the source code for the `attrbench` package, which provides a [Pytorch]()-based implementation of several metrics for the evaluation of feature attribution maps and methods. The package is currently mainly focused on image data, but we plan to generalize the code to work for any type of data.

## Installation
To install the project, run:
```bash
pip install attrbench
```

## Available metrics
- Deletion
- Insertion
- IROF
- IIOF
- Impact Coverage
- Impact Score
- Infidelity
- Max-Sensitivity
- Minimal Subset
- Sensitivity-n
- Seg-Sensitivity-n

## Future work
We are mainly working in 3 general directions for the near future:
- Generalization of the benchmark to other data modalities than images.
- Optimization of GPU usage.
- Inclusion of new metrics
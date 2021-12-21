# AttrBench: Metrics for Feature Attribution Techniques
This repository contains the source code for the `attrbench` package, which provides a [Pytorch]()-based implementation of several metrics for the evaluation of feature attribution maps and methods. The package is currently mainly focused on image data, but we plan to generalize the code to work for any type of data.

## Installation
TODO

## Usage
TODO

## Available metrics
| Name | Paper | Implemented |
| ---- | ----- | ----------- |
| Dataset masking accuracy | None | :heavy_check_mark: |
| Model masking accuracy | None | :heavy_check_mark: |
| Noise invariance | None | :heavy_check_mark: |
| Insertion curves | [Arxiv](https://arxiv.org/abs/1806.07421) | :x: |
| Deletion curves | [Arxiv](https://arxiv.org/abs/1806.07421) | :heavy_check_mark: |
| Sensitivity-n | [Arxiv](https://arxiv.org/abs/1711.06104) | :heavy_check_mark: |
| Infidelity | [Arxiv](https://arxiv.org/abs/1901.09392) | :heavy_check_mark: |
| Max-sensitivity | [Arxiv](https://arxiv.org/abs/1901.09392) | :heavy_check_mark: |
| Impact Score | [Arxiv](https://arxiv.org/abs/1910.07387) | :construction: |
| Impact Coverage | [Arxiv](https://arxiv.org/abs/1910.07387) | :construction: |
| Sanity Checks | [Arxiv](https://arxiv.org/abs/1810.03292) | :construction: |
| AIC | [Arxiv](https://arxiv.org/abs/1906.02825) | :x: |
| SIC | [Arxiv](https://arxiv.org/abs/1906.02825) | :x: |
| BAM | [Arxiv](https://arxiv.org/abs/1907.09701) | :x: |

## Future work
We are mainly working in 2 general directions for the near future:
- Generalization of the benchmark to other data modalities than images.
- Optimization of GPU usage.
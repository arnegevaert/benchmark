# ML Interpretability Benchmark
## Implemented techniques
- Sensitivity analysis (Simonyan et al. 2014)
- Input X Gradient (Kindermans et al. 2016)
- Integrated gradients (Sundararajan et al. 2017)

- Anchors (Ribeiro et al. 2018)
- Sufficient Input Subsets (Carter et al. 2018)

## Implemented performance measures
- (Robustness)
- FI: Relevance

## TODO
- RB: Generality
- IB: Sparsity (change as few features as possible on tabular data)

## Personal notes
Anchors package (anchor_exp) contains bugs:
  - anchor_tabular: uses old version of sklearn with old API for OneHotEncoder.
=> Maybe fork and debug?

## TODO
- SIS: keeps running. Not sure if infinite loop or inefficient code.

# ML Interpretability Benchmark
Benchmark for ML interpretability techniques.
<hr/>
## Techniques

#### Feature Attribution (FA) techniques
| Name | Paper | Source | Model requirements | Data type | Implemented |
| ---- | ----- | ------ | ------------------ | --------- | ----------- |
| Gradient | [link](https://arxiv.org/abs/1312.6034) | [Captum](https://captum.ai/api/saliency.html) | Differentiable | Any | :heavy_check_mark: |
| InputXGradient | [link](https://arxiv.org/abs/1611.07270) | [Captum](https://captum.ai/api/input_x_gradient.html) | Differentiable | Any | :heavy_check_mark: |
| IntegratedGradients | [link](http://arxiv.org/abs/1703.01365) | [Captum](https://captum.ai/api/integrated_gradients.html) | Differentiable | Any | :heavy_check_mark: |
| DeepLIFT<sup>1</sup> | [link](https://arxiv.org/abs/1704.02685) | [Captum **(WIP)**](https://captum.ai/api/deep_lift.html) | DNN | Image | :x: |
| LIME | [link](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | [Github](https://github.com/marcotcr/lime) | None | Any | :x: |
| SHAP | [link](https://arxiv.org/abs/1705.07874) | [Github](https://github.com/slundberg/shap) | None | Any | :x: |
| CGI | [link](https://arxiv.org/abs/1905.12152) | None | Differentiable | Any | :x: |
| XRAI | [link](https://arxiv.org/abs/1906.02825) | [Github](https://github.com/PAIR-code/saliency) | Differentiable | Image | :x: |
| CAM | [link](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) | [Github](https://github.com/zhoubolei/CAM) | CNN (GAP layer) | Image | :x: |
| Grad-CAM | [link](https://arxiv.org/abs/1610.02391) | [Github](https://github.com/ramprs/grad-cam/) | CNN | Image | :x: |
| Guided Grad-CAM | [link](https://arxiv.org/abs/1610.02391) | [Github](https://github.com/ramprs/grad-cam/) | CNN | Image | :x: |

<sup>1</sup>: DeepLIFT has 3 assignment rules (Linear, RC, RS), so should be split in DeepLIFT-{Linear,RC,RS}.

#### Rule-based (RB) techniques
| Name | Paper | Source | Model requirements | Data type | Implemented |
| ---- | ----- | ------ | ------------------ | --------- | ----------- |

#### Counterfactual (CF) techniques
| Name | Paper | Source | Model requirements | Data type | Implemented |
| ---- | ----- | ------ | ------------------ | --------- | ----------- |


## Metrics
We provide a list of metrics that can be used to assess the quality of explainability techniques.
:heavy_check_mark: means the metric is already implemented, :x: means it is not.
#### Accuracy
Applicable to FA techniques.
Given (partial) ground truth about feature importances, FA method should match this ground truth as well as possible. Ways to measure this:

- **Using adversarial examples :x::**
    - Generate (localized https://arxiv.org/abs/1801.02608) adversarial noise to change the output of the model
    - Calculate feature attribution on adversarial example
    - Feature attribution (change in feature attribution?) should correlate with adversarial noise
- **Using synthetic data :x::**
    - Generate dataset where parts of input cause the output label, other parts are noise
    - Feature attribution should match these parts
- **Using model masking :x::**
    - Create a model on an existing dataset that only looks at part of the input
    - Train it to predict a property that depends on ALL the remaining features (e.g. Y=1 iff ALL remaining features > 0.5)
    - Attribution should match exactly with the unmasked parts of the input, as the method by definition doesn't look at the masked parts, and needs ALL of the unmasked parts to accurately predict.

#### Robustness
Applicable to FA techniques. Noise/constant shifts in input that don't change the model output should not change the attribution. Might also be applicable to RB/CF techniques.

- Add uniform noise to image and measure change in attribution :heavy_check_mark:
- Add constant mean shift to image and measure change in attribution :heavy_check_mark:

#### Relevance
Features that are deemed important (i.e. have a large attribution) should have a large influence on the model output.

- Mask important features and measure output confidence decrease :heavy_check_mark:
- Overlay important feature values on random background (or different sample) and measure confidence increase :x:

#### Model dependence
Applicable to FA techniques. Explanations for a random or trained model should differ significantly. See also: [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292).

- Get explanation for randomized model vs trained model and measure difference
- Get explanation for trained model vs model trained on random labels and measure difference

#### Stability
Applicable to all techniques. Output should change as little as possible if hyperparameters are varied.

- Get explanations using different (realistic) hyperparameter settings and measure variance in explanations :x:.

#### Scalability
Applicable to all techniques.

- **w.r.t. dimensionality:** Measure execution time (assess theoretical complexity) per sample for varying dimensionality.
- **w.r.t. amount of explanations:** Measure average execution time per sample for varying amount of samples.


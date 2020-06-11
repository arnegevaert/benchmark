# ML Interpretability Benchmark
Benchmark for ML interpretability techniques.

## Techniques

#### Feature Attribution (FA) techniques
- Implemented: :heavy_check_mark:
- Work in progress: :construction:
- Not implemented: :x:

| Name | Paper | Source | Model requirements | Data type | Implemented |
| ---- | ----- | ------ | ------------------ | --------- | ----------- |
| Gradient | [link](https://arxiv.org/abs/1312.6034) | [Captum](https://captum.ai/api/saliency.html) | Differentiable | Any | :heavy_check_mark: |
| InputXGradient | [link](https://arxiv.org/abs/1611.07270) | [Captum](https://captum.ai/api/input_x_gradient.html) | Differentiable | Any | :heavy_check_mark: |
| IntegratedGradients | [link](http://arxiv.org/abs/1703.01365) | [Captum](https://captum.ai/api/integrated_gradients.html) | Differentiable | Any | :heavy_check_mark: |
| Guided Backprop | [link](https://arxiv.org/abs/1412.6806) | [Captum](https://captum.ai/api/guided_backprop.html) | DNN | Image | :heavy_check_mark: |
| Guided Grad-CAM | [link](https://arxiv.org/abs/1610.02391) | [Captum](https://captum.ai/api/guided_grad_cam.html) | CNN | Image | :heavy_check_mark: |
| Deconvolution | [link](https://arxiv.org/abs/1311.2901) | [Captum](https://captum.ai/api/deconvolution.html) | CNN | Image | :heavy_check_mark: |
| Feature Ablation | None | [Captum](https://captum.ai/api/feature_ablation.html) | None | Any | :construction: |
| DeepLIFT<sup>1</sup> | [link](https://arxiv.org/abs/1704.02685) | [Captum **(UNR)**](https://captum.ai/api/deep_lift.html) | DNN | Image | :construction: |
| Occlusion | [link](https://arxiv.org/abs/1311.2901) | [Captum](https://captum.ai/api/occlusion.html) | none | Any | :construction: |
| LIME | [link](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) | [PyPi](https://pypi.org/project/lime/) | None | Any | :heavy_check_mark: |
| SHAP | [link](https://arxiv.org/abs/1705.07874) | [PyPi](https://pypi.org/project/shap/) | None | Any | :construction: |
| CGI | [link](https://arxiv.org/abs/1905.12152) | None | Differentiable | Any | :x: |
| XRAI | [link](https://arxiv.org/abs/1906.02825) | [Github](https://github.com/PAIR-code/saliency) | Differentiable | Image | :x: |
| CAM | [link](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) | [Github](https://github.com/zhoubolei/CAM) | CNN (GAP layer) | Image | :x: |
| Grad-CAM | [link](https://arxiv.org/abs/1610.02391) | [Github](https://github.com/ramprs/grad-cam/) | CNN | Image | :x: |
| Masking model | [link](https://arxiv.org/abs/1705.07857) | None | Differentiable | Image | :x: |
| SmoothGrad | [link](https://arxiv.org/abs/1706.03825) | None | Differentiable | Any | :x: |
| Expected Gradients | [link](https://arxiv.org/abs/1906.10670) | None | Differentiable | Any | :x: |

<sup>1</sup>: DeepLIFT has 3 assignment rules (Linear, RC, RS), so should be split in DeepLIFT-{Linear,RC,RS}.

## Metrics
We provide a list of metrics that can be used to assess the quality of explainability techniques.
:heavy_check_mark: means the metric is already implemented, :x: means it is not.

| Name | Paper | Implemented |
| ---- | ----- | ----------- |
| Dataset masking accuracy | None | :heavy_check_mark: |
| Model masking accuracy | None | :heavy_check_mark: |
| Noise invariance | None | :heavy_check_mark: |
| Simple Sensitivity | None | :heavy_check_mark: |
| Sensitivity-n | [Arxiv](https://arxiv.org/abs/1711.06104) | :heavy_check_mark: |
| Infidelity | [Arxiv](https://arxiv.org/abs/1901.09392) | :heavy_check_mark: |
| Max-sensitivity | [Arxiv](https://arxiv.org/abs/1901.09392) | :heavy_check_mark: |
| Impact Score | [Arxiv](https://arxiv.org/abs/1910.07387) | :construction: |
| Impact Coverage | [Arxiv](https://arxiv.org/abs/1910.07387) | :construction: |
| Sanity Checks | [Arxiv](https://arxiv.org/abs/1810.03292) | :construction: |
| AIC | [Arxiv](https://arxiv.org/abs/1906.02825) | :x: |
| SIC | [Arxiv](https://arxiv.org/abs/1906.02825) | :x: |
| BAM | [Arxiv](https://arxiv.org/abs/1907.09701) | :x: |






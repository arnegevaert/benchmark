from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import torch.nn as nn
import torch


# TODO using implementation from https://github.com/interpretml/interpret might be faster (implemented in C++ instead of python)
class LIME:
    def __init__(self, model: nn.Module, sample_shape, normalize=True, aggregation_fn=None):
        super(LIME, self).__init__(absolute=False, normalize=normalize, aggregation_fn=aggregation_fn)
        # TODO make this segmentation algorithm/parameters configurable
        self.segmentation = SegmentationAlgorithm("quickshift", kernel_size=5, max_dist=10, ratio=1)
        self.kernel_size = 0.25
        self.explainer = lime_image.LimeImageExplainer(self.kernel_size)
        self.n_samples = 1000
        self.model = model
        self.sample_shape = sample_shape

    def transform_grayscale(self, x):
        x = torch.tensor(x)  # [batch_size, rows, cols, 3]
        x = x.unsqueeze(1)  # [batch_size, 1, rows, cols, 3]
        x = x[:, :, :, :, 0]  # [batch_size, 1, rows, cols]
        return x

    def transform_rgb(self, x):
        x = torch.tensor(x)
        return x.transpose(3, 1)

    def __call__(self, x, target):
        if self.sample_shape[0] == 1:
            # Images are grayscale
            x = x.squeeze(dim=1)
            transform = self.transform_grayscale
        else:
            # Images are RGB
            x = x.transpose(1, 3)
            transform = self.transform_rgb
        result = []
        for i in range(x.shape[0]):
            image = x[i].detach().numpy()
            lime_explanation = self.explainer.explain_instance(image, lambda x: self.model(transform(x)).detach().numpy(),
                                                               segmentation_fn=self.segmentation)
            scores = lime_explanation.local_exp[target[i].item()]
            segments = torch.tensor(lime_explanation.segments).unsqueeze(0)
            # LIME attributions are always pixel-level
            explanation = torch.zeros(1, self.sample_shape[1], self.sample_shape[2])
            for segment, score in scores:
                explanation[segments == segment] = score
            explanation = explanation.unsqueeze(0)
            result.append(explanation)
        result = torch.cat(result, 0)
        return result

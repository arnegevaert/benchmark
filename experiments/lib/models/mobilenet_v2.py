import torch
import torchvision
import torch.nn as nn


class Mobilenet_v2(nn.Module):
    def __init__(self, num_classes, params_loc=None):
        super().__init__()
        tmp_model = torchvision.models.mobilenet_v2()
        self.features = tmp_model.features
        self.classifier = tmp_model.classifier
        num_ftrs = self.classifier[1].in_features
        self.classifier[1] = nn.Linear(num_ftrs, num_classes)

        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        return self.features[18][0]

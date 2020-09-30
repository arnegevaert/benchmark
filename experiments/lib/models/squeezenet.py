import torch
import torchvision
import torch.nn as nn


_versions = ["squeezenet1_0", "squeezenet1_1"]


class Squeezenet(nn.Module):
    def __init__(self, version, num_classes, params_loc=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
        if version not in _versions:
            raise NotImplementedError("Version not supported")
        fn = getattr(torchvision.models, version)
        base_model = fn()
        self.features = base_model.features

        # Taken from Squeezenet source
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def get_last_conv_layer(self) -> nn.Module:
        return self.classifier[1]  # final_conv in constructor

import torch.nn as nn
from torch.nn import functional as F
import torch

class BasicCNN(nn.Module):
    """
    Basic convolutional network for MNIST
    """
    def __init__(self, num_classes, params_loc=None):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if params_loc:
            # map_location allows taking a model trained on GPU and loading it on CPU
            # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
        if x.dtype != torch.float32:
            x = x.float()

        relu = nn.ReLU()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def get_last_conv_layer(self):
        return self.conv2
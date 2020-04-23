import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path, getenv
from util.models import ConvolutionalNetworkModel
import numpy as np

DEVICE = torch.device(getenv("TORCH_DEVICE", "cpu"))


class Net(nn.Module):
    def __init__(self, output_logits):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.output_logits = output_logits

    def get_logits(self, x):
        relu = nn.ReLU()
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        return self.fc2(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        if self.output_logits:
            return logits
        return F.softmax(logits, dim=1)


class MNISTCNN(ConvolutionalNetworkModel):
    def __init__(self, output_logits=False):
        super().__init__()
        self.net = Net(output_logits).to(DEVICE)
        params_loc = path.join(path.dirname(__file__), "saved_models", "mnist_cnn.pth")
        if not path.exists(params_loc):
            raise FileNotFoundError(f"{params_loc} does not exist. "
                                    f"Use the train_mnist_cnn.py script to train and save weights.")
        # map_location allows taking a model trained on GPU and loading it on CPU
        # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
        self.net.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))
        self.output_logits = output_logits

    def predict(self, x):
        self.net.eval()
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        return self.net(x)

    def get_conv_net(self) -> nn.Module:
        return self.net

    def get_last_conv_layer(self) -> nn.Module:
        return self.net.conv2

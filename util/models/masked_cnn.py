import torch.nn as nn
from util.models import ConvolutionalNetworkModel
from masking_accuracy import MaskedInputLayer
import torch
import torch.nn.functional as F
import pickle as pkl


class Net(nn.Module):
    def __init__(self, sample_shape, mask_radius, mask_value):
        super(Net, self).__init__()
        self.masked_input_layer = MaskedInputLayer(sample_shape, mask_radius, mask_value)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def get_logits(self, x):
        relu = nn.ReLU()
        x = self.masked_input_layer(x)
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = relu(x)
        x = self.dropout3(x)
        return self.fc3(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        return F.softmax(logits, dim=1)

    def mask(self, x):
        return self.masked_input_layer(x)


# TODO automatic loading of parameters (see MNISTCNN)
class MaskedCNN(ConvolutionalNetworkModel):
    def __init__(self, sample_shape, mask_radius, mask_value):
        super().__init__()
        self.net = Net(sample_shape, mask_radius, mask_value)
        self.sample_shape = sample_shape
        self.mask_radius = mask_radius
        self.mask_value = mask_value

    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def get_conv_net(self) -> nn.Module:
        return self.net

    def get_last_conv_layer(self) -> nn.Module:
        return self.net.conv2

    def get_mask(self):
        return self.net.masked_input_layer.mask

    def mask(self, x):
        with torch.no_grad():
            return self.net.mask(x)

    def save(self, location):
        d = {
            "params": {
                "sample_shape": self.sample_shape,
                "mask_radius": self.mask_radius,
                "mask_value": self.mask_value
            },
            "state_dict": self.net.state_dict()
        }
        pkl.dump(d, open(location, "wb"))

    @staticmethod
    def load(location):
        d = pkl.load(open(location, "rb"))
        cnn = MaskedCNN(**d["params"])
        cnn.net.load_state_dict(d["state_dict"])
        return cnn

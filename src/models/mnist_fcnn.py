import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path, getenv
from models.model import ConvolutionalNetworkModel, Model

DEVICE = torch.device(getenv("TORCH_DEVICE", "cpu"))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(28*28,50)
        self.dense2 = nn.Linear(50,50)
        self.out = nn.Linear(50,10)

    def get_logits(self, x):
        relu = nn.ReLU()
        x= torch.flatten(x, 1)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        return self.out(x)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        logits = self.get_logits(x)
        return F.softmax(logits, dim=1)


class MNISTFCNN(Model):
    def __init__(self):
        super().__init__()
        self.net = Net().to(DEVICE)
        params_loc = path.join(path.dirname(__file__), "saved_models", "mnist_fcnn.pth")
        if not path.exists(params_loc):
            raise FileNotFoundError(f"{params_loc} does not exist. "
                                    f"Use the train_mnist_cnn.py script to train and save weights.")
        # map_location allows taking a model trained on GPU and loading it on CPU
        # without it, a model trained on GPU will be loaded in GPU even if DEVICE is CPU
        self.net.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def get_conv_net(self) -> nn.Module:
        return self.net

    def get_last_conv_layer(self) -> nn.Module:
        return self.net.dense2

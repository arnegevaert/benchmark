import torch
import torch.nn as nn


class BasicMLP(nn.Module):
    def __init__(self, num_classes, params_loc=None):
        super(BasicMLP, self).__init__()
        self.dense1 = nn.Linear(28*28, 50)
        self.dense2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, num_classes)
        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        relu = nn.ReLU()
        x= torch.flatten(x, 1)
        x = self.dense1(x)
        x = relu(x)
        x = self.dense2(x)
        x = relu(x)
        return self.out(x)

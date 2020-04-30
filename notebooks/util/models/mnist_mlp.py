import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTMLP(nn.Module):
    def __init__(self, output_logits, params_loc=None):
        super(MNISTMLP, self).__init__()
        self.output_logits = output_logits
        self.dense1 = nn.Linear(28*28, 50)
        self.dense2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 10)
        if params_loc:
            self.load_state_dict(torch.load(params_loc, map_location=lambda storage, loc: storage))

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

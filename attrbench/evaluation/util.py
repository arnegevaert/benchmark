import torch


def logit_softmax(logits):
    lse = torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True) - torch.exp(logits))
    return logits - lse

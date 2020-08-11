import torch


# TODO does adding a constant here have an influence on the outcome? Numerical errors
def logit_softmax(logits):
    lse = torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True) - torch.exp(logits))
    return logits - lse

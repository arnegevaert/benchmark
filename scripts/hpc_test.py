import torch
device = torch.device("cuda")

x = torch.ones(10)
x = x.to(device)

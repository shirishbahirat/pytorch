import torch


x = torch.rand((2, 3))

torch.einsum('ij->ji', x)

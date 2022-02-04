import torch


x = torch.rand((2, 3))

torch.einsum('ij->ji', x)

print(x, torch.einsum('ij->ji', x))

# summation

print(torch.einsum('ij->', x))

# column sum of

print(torch.einsum('ij->j', x))

# row sum

print(torch.einsum('ij->i', x))

v = torch.rand((1, 3))
torch.einsum('ij,kj->ik', x, v)

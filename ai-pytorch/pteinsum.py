import torch


x = torch.rand((2, 3))

torch.einsum('ij->ji', x)

print(x, torch.einsum('ij->ji', x))

# summation

print(torch.einsum('ij->', x))

# column sum of

print(torch.einsum('ij->', x))

# row sum

print(torch.einsum('ij->', x))

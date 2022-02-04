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

# Matrix vector multiplication
v = torch.rand((1, 3))
print(torch.einsum('ij,kj->ik', x, v))


# Matrix multiplication
print(torch.einsum('ij,kj->ik', x, x))

# dot product
print(torch.einsum('i,i->', x[0], x[0]))

# Hadamard product (elementwise multiplication)
print(torch.einsum('ij,ij->ij', x, x))

# outer product
a = torch.rand((3))
b = torch.rand((5))

print(torch.einsum('i,j->ij', a, b))

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

a = torch.tensor(4., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

print(a)
print(b)

y = a**2 + b

print(y)

y.backward()

print(a.grad)
print(b.grad)

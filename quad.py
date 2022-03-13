import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

a = torch.tensor(6., requires_grad=True)
b = torch.tensor(7., requires_grad=True)
c = torch.tensor(5., requires_grad=True)
d = torch.tensor(2., requires_grad=True)

print(a)
print(b)
print(c)
print(d)


y = a**3 + b**2 + c**1 + d

print(y)

y.backward()

print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

print(x)
print(w)
print(b)


y = w * x + b

print(y)

y.backward()

print(x.grad)    # x.grad = 2
print(w.grad)    # w.grad = 1
print(b.grad)    # b.grad = 1

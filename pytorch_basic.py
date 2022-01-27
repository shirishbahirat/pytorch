import torch
import torchvision
from torch.autograd import Variable
import numpy as np


def f(x):
    return x ** 2 + 2 * x + 6


np_x = np.array([4.0])
x = torch.from_numpy(np_x).requires_grad_(True)
y = f(x)
print(y)

y.backward()
x.grad
print(x.grad)

np_x = np.array([5.0])
x = torch.from_numpy(np_x).requires_grad_(True)
y = f(x)
print(y)

y.backward()
x.grad
print(x.grad)

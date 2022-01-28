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


np_x = np.array([10.])
np_x = np_x.transpose()
x = torch.from_numpy(np_x).requires_grad_(True)
y = f(x)
print(y)

y.backward()
x.grad
print(x.grad)


a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

print(a, b)

Q = 3 * a**3 - b**2
print(Q)
external_grad = torch.tensor([1., 1.])

Q.backward(gradient=external_grad)

print(a.grad)
print(b.grad)

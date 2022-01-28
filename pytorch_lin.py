import torch
import torchvision
from torch.autograd import Variable
import numpy as np
from autograd import grad
import autograd.numpy.random as npr

N = 100
x = np.linspace(0, 10, N)

t = 4 * x + 10 + npr.normal(0, 2, x.shape[0])

w = npr.normal(0, 1)
b = npr.normal(0, 1)
params = {'w': w, 'b': b}  # One option: aggregate parameters in a dictionary


def mode(x):
    return x ** 2 + 2 * x + 6


def cost(x):
    return (1 / N) * np.sum(((2 * x + 2)**2) * 0.5)


grad_cost = grad(cost)

print(mode(x))


num_epochs = 1000  # Number of epochs of training
alpha = 0.01       # Learning rate

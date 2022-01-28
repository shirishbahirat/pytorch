import torch
import torchvision
from torch.autograd import Variable
import numpy as np
from autograd import grad

N = 100
x = np.linspace(0, 10, N)


def mode(x):
    return x ** 2 + 2 * x + 6


def cost(x):
    return (1 / N) * np.sum(((2 * x + 2)**2) * 0.5)


grad_cost = grad(cost)

print(mode(x))


num_epochs = 1000  # Number of epochs of training
alpha = 0.01       # Learning rate

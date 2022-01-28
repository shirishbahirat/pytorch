import torch
import torchvision
from torch.autograd import Variable
import numpy as np


def mode(x):
    return x ** 2 + 2 * x + 6


def cost(x):
    return ((2 * x + 2)**2) * 0.5


grad_cost = grad(cost)

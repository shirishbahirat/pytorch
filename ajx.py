import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

dataset = MNIST(root='data/', download=True)

print(len(dataset))

test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset))

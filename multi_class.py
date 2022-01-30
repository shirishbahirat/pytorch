import numpy as np
import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, input, hidden, classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

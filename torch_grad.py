import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


x_data = Variable(torch.tensor([[1.], [2.], [3.]]))
y_data = Variable(torch.tensor([[2.], [4.], [6.]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model()

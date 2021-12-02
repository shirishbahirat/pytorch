import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


w = Variable(torch.tensor([1.0]), requires_grad=True)
w_list = []
mse_list = []

x_data = Variable(torch.tensor([1., 2., 3.]))
y_data = Variable(torch.tensor([2., 4., 6.]))


def forward(x):
    return x * w


def loss(x, y):
    err = forward(x) - y
    return (err * err)


def gradient(x, y):
    return 2 * x * (x * w - y)


print("predict (before training):", 4, forward(4))


for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\t grad: ", x_val, y_val, w.grad.data[0])
        w.grad.data.zero_()
    print("progress: ", epoch, l.data[0])


print("predict (after training):", 4, forward(4))

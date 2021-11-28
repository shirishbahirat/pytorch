import numpy as np
import matplotlib.pyplot as plt


w = 1.0
w_list = []
mse_list = []

x_data = [1., 2., 3.]
y_data = [2., 4., 6.]


def forward(x):
    return x * w


def loss(x, y):
    err = forward(x) - y
    return (err * err)


def gradient(x, y):
    return 2 * x * (x * w - y)


print("predict (before training):", 4, forward(4))

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\t grad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)
    print("progress: ", epoch, "w:", w, "loss:", l)

print("predict (after training):", 4, forward(4))

import numpy as np
import matplotlib.pyplot as plt

w1 = 1.0
w2 = 2.0
w1_list = []
w2_list = []
mse_list = []

x_data = [1., 2., 3.]
y_data = [0., 0., 0.]

y_data[0] = x_data[0] * x_data[0] * 4 + x_data[0] * 3 + 10
y_data[1] = x_data[1] * x_data[1] * 4 + x_data[1] * 3 + 10
y_data[2] = x_data[2] * x_data[2] * 4 + x_data[2] * 3 + 10


def forward(x):
    return x * x * w1 + x * w2 + 10


def loss(x, y):
    err = forward(x) - y
    return (err * err)


def gradient_w1(x, y):
    return 2 * (forward(x) - y) * w1 * x


def gradient_w2(x, y):
    return 2 * (forward(x) - y) * w2


print("predict (before training):", 4, forward(3))

for epoch in range(50):
    for x_val, y_val in zip(x_data, y_data):
        grad1 = gradient_w1(x_val, y_val)
        w1 = w1 - 0.01 * grad1
        grad2 = gradient_w2(x_val, y_val)
        w2 = w2 - 0.01 * grad2
        print("\t grad: ", x_val, y_val, grad1, grad2)
        l = loss(x_val, y_val)
    print("progress: ", epoch, "w:", w1, w2, "loss:", l)

print("predict (after training):", 4, forward(4))

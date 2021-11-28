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


for w in np.arange(0., 4.1, .1):
    print ("w: ", w)
    mse = 0
    lsum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val)
        l = loss(x_val, y_val)
        lsum += l
        print("\t", x_val, y_val, y_pred, l)
        mse = lsum / 3
    print("MSE:", mse)

    w_list.append(w)
    mse_list.append(mse)


plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()

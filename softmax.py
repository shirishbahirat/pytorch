import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2., 1., .1])

print(softmax(x))

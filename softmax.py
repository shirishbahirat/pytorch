import numpy as np
import torch


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2., 1., .1])

print(softmax(x))

x = torch.tensor(x)
print(torch.softmax(x, dim=0))

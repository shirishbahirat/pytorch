import numpy as np
import torch
from torch import nn


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2., 1., .1])

print(softmax(x))

x = torch.tensor(x)
print(torch.softmax(x, dim=0))


def cross_entropy(actual, predicted):
    return -np.sum(actual * np.log(predicted))


y = np.array([1, 0, 0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)

print(f'loss1 numpy: {l1:.4f}')
print(f'loss2 numpy: {l2:.4f}')


loss = nn.CrossEntropyLoss()
Y = torch.tensor([1])
Y_pred_good = torch.tensor([[0.7, 0.2, 0.1]])
Y_pred_bad = torch.tensor([[0.1, 0.3, 0.6]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'loss1 numpy: {l1.item():.4f}')
print(f'loss2 numpy: {l2.item():.4f}')

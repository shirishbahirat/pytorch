import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from jax import random


import numpy as onp
import jax.numpy as np

key = random.PRNGKey(1)
x = random.uniform(key, (1000, 1000))

time y = onp.dot(x, x)
time y = np.dot(x, x)
time y = np.dot(x, x).block_until_ready()


dataset = MNIST(root='data/', download=True)

print(len(dataset))

test_dataset = MNIST(root='data/', train=False)
print(len(test_dataset))

dataset = MNIST(root='data/',
                train=True,
                transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label)

print(img_tensor[0, 10:15, 10:15])
print(torch.max(img_tensor), torch.min(img_tensor))

train_ds, val_ds = random_split(dataset, [50000, 10000])


batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28 * 28
num_classes = 10


import numpy as np


def predict(params, inputs):
    for W, b in params:
        output = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs


def mse_loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return np.sum((preds - targets) ** 2)


import jax.numpy as np
from jax import grad, jit, vmap, pmap


def predict(params, inputs):
    for W, b in params:
        output = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs


def mse_loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return np.sum((preds - targets) ** 2)


gradient_fun = jit(grad(mse_loss))
perexample_grads = jit(vmap(grad(mse_loss), in_axes=(None, 0)))
parallel_grads = jit(pmap(grad(mse_loss), in_axes=(None, 0)))

import autograd.numpy as np
from autograd import grad


def tanh(x):
    y = np.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)


grad_tanh = grad(tanh)

grad_tanh(1.0)

(tanh(1.0001) - tanh(0.9999)) / 0.0002

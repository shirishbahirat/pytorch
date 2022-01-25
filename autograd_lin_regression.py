import autograd.numpy as np  # Import wrapped NumPy from Autograd
import autograd.numpy.random as npr  # For convenient access to numpy.random
from autograd import grad  # To compute gradients

import matplotlib.pyplot as plt  # For plotting

# In our synthetic data, we have w = 4 and b = 10
N = 100  # Number of training data points
x = np.linspace(0, 10, N)
t = 4 * x + 10 + npr.normal(0, 2, x.shape[0])
plt.plot(x, t, 'r.')
plt.show()

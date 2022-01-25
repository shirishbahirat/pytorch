import autograd.numpy as np  # Import wrapped NumPy from Autograd
import autograd.numpy.random as npr  # For convenient access to numpy.random
from autograd import grad  # To compute gradients

import matplotlib.pyplot as plt  # For plotting


# Generate synthetic data
N = 100  # Number of data points
x = np.linspace(-3, 3, N)  # Generate N values linearly-spaced between -3 and 3
t = x ** 4 - 10 * x ** 2 + 10 * x + npr.normal(0, 4, x.shape[0])  # Generate corresponding targets

M = 4  # Degree of polynomial to fit to the data (this is a hyperparameter)
feature_matrix = np.array([[item ** i for i in range(M + 1)] for item in x])  # Construct a feature matrix
W = npr.randn(feature_matrix.shape[-1])


def cost(W):
    y = np.dot(feature_matrix, W)
    return (1.0 / N) * np.sum(0.5 * np.square(y - t))


# Compute the gradient of the cost function using Autograd
cost_grad = grad(cost)

num_epochs = 10000
learning_rate = 0.001

# Manually implement gradient descent
for i in range(num_epochs):
    W = W - learning_rate * cost_grad(W)

# Print the final learned parameters.
print(W)

plt.plot(x, t, 'r.')
plt.plot(x, np.dot(feature_matrix, W), 'b-')
plt.show()

import autograd.numpy as np  # Import wrapped NumPy from Autograd
import autograd.numpy.random as npr  # For convenient access to numpy.random
from autograd import grad  # To compute gradients

import matplotlib.pyplot as plt  # For plotting

# In our synthetic data, we have w = 4 and b = 10
N = 100  # Number of training data points
x = np.linspace(0, 10, N)
t = 4 * x + 10 + npr.normal(0, 2, x.shape[0])


# Initialize random parameters
w = npr.normal(0, 1)
b = npr.normal(0, 1)
params = {'w': w, 'b': b}  # One option: aggregate parameters in a dictionary


def cost(params):
    y = params['w'] * x + params['b']
    return (1 / N) * np.sum(0.5 * np.square(y - t))


# Find the gradient of the cost function using Autograd
grad_cost = grad(cost)

num_epochs = 1000  # Number of epochs of training
alpha = 0.01       # Learning rate

for i in range(num_epochs):
    # Evaluate the gradient of the current parameters stored in params
    cost_params = grad_cost(params)

    # Update parameters w and b
    params['w'] = params['w'] - alpha * cost_params['w']
    params['b'] = params['b'] - alpha * cost_params['b']

print(params)

# Plot the training data again, together with the line defined by y = wx + b
# where w and b are our final learned parameters
plt.plot(x, t, 'r.')
plt.plot([0, 10], [params['b'], params['w'] * 10 + params['b']], 'b-')
plt.show()

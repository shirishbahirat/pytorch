import jax
from jax import numpy as jnp, random

import numpy as np

m = jnp.identity(4)
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])

print(n)
print(m)

x = jnp.dot(n, m).block_until_ready()

print(x)

m = jnp.ones((4, 4))
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])

print(n)
print(m)

x = jnp.dot(n, m).block_until_ready()

print(x)

a = np.random.normal(size=(4, 4))

b = jnp.dot(a, m)

print(b)

b = jax.device_put(b)

q = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
np.array(q)

print(q)

w = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
updated = jax.ops.index_update(w, (0, 0), 3.0)
print("w: \n", w)
print("updated: \n", updated)

x = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])
jax.ops.index_update(x, jax.ops.index[0, :], 3.0)  #  Same as x[O,:] = 3.0 in NumPy.


key = random.PRNGKey(0)


key = random.PRNGKey(0)


def f(x):
    return jnp.dot(x.T, x) / 2.0


v = jnp.ones((4,))
f(v)

v = random.normal(key, (4,))
print("Original v:")
print(v)
print("Gradient of f taken at point v")
print(jax.grad(f)(v))  # should be equal to v !


def f(x):
    return jnp.multiply(x, x) / 2.0


x = random.normal(key, (5,))
v = jnp.ones(5)
print("(x,f(x))")
print((x, f(x)))
print("jax.jvp(f, (x,),(v,))")
print(jax.jvp(f, (x,), (v,)))


mat = random.normal(key, (15, 10))
batched_x = random.normal(key, (5, 10))  # Batching on first dimension
single = random.normal(key, (10,))


def apply_matrix(v):
    return jnp.dot(mat, v)


print("Single apply shape: ", apply_matrix(single).shape)
print("Batched example shape: ", jax.vmap(apply_matrix)(batched_x).shape)


key = random.PRNGKey(0)

# Create the predict function from a set of parameters


def make_predict(W, b):
    def predict(x):
        return jnp.dot(W, x) + b
    return predict

# Create the loss from the data points set


def make_mse(x_batched, y_batched):
    def mse(W, b):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            y_pred = make_predict(W, b)(x)
            return jnp.inner(y - y_pred, y - y_pred) / 2.0
        # We vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
    return jax.jit(mse)  # And finally we jit the result.



# Set problem dimensions
nsamples = 20
xdim = 10
ydim = 5

# Generate random ground truth W and b
k1, k2 = random.split(key)
W = random.normal(k1, (ydim, xdim))
b = random.normal(k2, (ydim,))
true_predict = make_predict(W, b)

# Generate samples with additional noise
ksample, knoise = random.split(k1)
x_samples = random.normal(ksample, (nsamples, xdim))
y_samples = jax.vmap(true_predict)(x_samples) + 0.1 * random.normal(knoise, (nsamples, ydim))

# Generate MSE for our samples
mse = make_mse(x_samples, y_samples)


# Initialize estimated W and b with zeros.
What = jnp.zeros_like(W)
bhat = jnp.zeros_like(b)

alpha = 0.3  # Gradient step size
print('Loss for "true" W,b: ', mse(W, b))
for i in range(101):
    # We perform one gradient update
    What, bhat = What - alpha * jax.grad(mse, 0)(What, bhat), bhat - alpha * jax.grad(mse, 1)(What, bhat)
    if (i % 5 == 0):
        print("Loss step {}: ".format(i), mse(What, bhat))


key = random.PRNGKey(0)

# Create the predict function from a set of parameters


def make_predict_pytree(params):
    def predict(x):
        return jnp.dot(params['W'], x) + params['b']
    return predict

# Create the loss from the data points set


def make_mse_pytree(x_batched, y_batched):
    def mse(params):
        # Define the squared loss for a single pair (x,y)
        def squared_error(x, y):
            y_pred = make_predict_pytree(params)(x)
            return jnp.inner(y - y_pred, y - y_pred) / 2.0
        # We vectorize the previous to compute the average of the loss on all samples.
        return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)
    return jax.jit(mse)  # And finally we jit the result.


# Generate MSE for our samples
mse_pytree = make_mse_pytree(x_samples, y_samples)

# Initialize estimated W and b with zeros.
params = {'W': jnp.zeros_like(W), 'b': jnp.zeros_like(b)}

jax.grad(mse_pytree)(params)


alpha = 0.3  # Gradient step size
print('Loss for "true" W,b: ', mse_pytree({'W': W, 'b': b}))
for i in range(101):
    # We perform one gradient update
    params = jax.tree_multimap(lambda old, grad: old - alpha * grad, params, jax.grad(mse_pytree)(params))
    if (i % 5 == 0):
        print("Loss step {}: ".format(i), mse_pytree(params))

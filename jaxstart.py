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

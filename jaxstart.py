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

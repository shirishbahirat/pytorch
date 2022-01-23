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

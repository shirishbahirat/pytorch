import jax
from jax import numpy as jnp, random

import numpy as np

m = jnp.ones((4, 4))  # We're generating one 4 by 4 matrix filled with ones.
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])  # An explicit 2 by 4 array
print(m)

d = jnp.dot(n, m).block_until_ready()
print(d)

x = np.random.normal(size=(4, 4))  # Creating one standard NumPy array instance
y = jnp.dot(x, m)
print(y)

x = np.random.normal(size=(4, 4))
x = jax.device_put(x)
print(x)

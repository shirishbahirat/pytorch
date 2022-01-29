import jax
from jax import numpy as jnp, random

import numpy as np

m = jnp.ones((4, 4))  # We're generating one 4 by 4 matrix filled with ones.
n = jnp.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0]])  # An explicit 2 by 4 array
print(m)

jnp.dot(n, m).block_until_ready()

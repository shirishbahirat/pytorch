import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten


params = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0, 3.0)]
flat_params, unflatten_func = flatten(params)
print('Flattened: {}'.format(flat_params))
print('Unflattened: {}'.format(unflatten_func(flat_params)))

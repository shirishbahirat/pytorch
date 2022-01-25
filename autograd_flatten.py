import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten


params = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0, 3.0)]
flat_params, unflatten_func = flatten(params)
print('Flattened: {}'.format(flat_params))
print('Unflattened: {}'.format(unflatten_func(flat_params)))

params = [npr.randn(3, 3), npr.randn(4, 4), npr.randn(3, 3)]
flat_params, unflatten_func = flatten(params)
print('Flattened: {}'.format(flat_params))
print('Unflattened: {}'.format(unflatten_func(flat_params)))

params = {'weights': [1.0, 2.0, 3.0, 4.0], 'biases': [1.0, 2.0]}
flat_params, unflatten_func = flatten(params)
print('Flattened: {}'.format(flat_params))
print('Unflattened: {}'.format(unflatten_func(flat_params)))

params = {'layer1': {'weights': [1.0, 2.0, 3.0, 4.0], 'biases': [1.0, 2.0]}, 'layer2': {'weights': [5.0, 6.0, 7.0, 8.0], 'biases': [6.0, 7.0]}}
flat_params, unflatten_func = flatten(params)
print('Flattened: {}'.format(flat_params))
print('Unflattened: {}'.format(unflatten_func(flat_params)))

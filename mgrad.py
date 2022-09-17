import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 3*x**2 - 4*x + 5

#6*x - 4

print(f(3.0))

xs = np.arange(-5, 7, 0.25)
ys = f(xs)


h = 0.000001
x = 3
print((f(x + h) - f(x))/h)

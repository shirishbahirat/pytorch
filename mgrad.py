import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 3*x**2 - 4*x + 5

#6*x - 4
#6*(2/3) - 4 = 0

print(f(3.0))

xs = np.arange(-5, 7, 0.25)
ys = f(xs)


h = 0.000001
x = 3
print((f(x + h) - f(x))/h)

h = 0.000001
x = 2/3
print((f(x + h) - f(x))/h)


a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

def g(a,b,c):
  return a*b + c


print((g(a,b,c + h) - g(a,b,c))/h)
print((g(a+h,b,c) - g(a,b,c))/h)
print((g(a,b+h,c) - g(a,b,c))/h)

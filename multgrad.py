import math

h = 0.0001

r = 3
theta = 3


def x(r, theta):
    return r*math.sin(theta)


def y(r, theta):
    return r*math.cos(theta)

a = x(r, theta)
b = y(r, theta)

a1 = x(r+h, theta)
b1 = y(r+h, theta)


def g(a,b):
    return a**2 * b**3


w1 = g(a, b)
w2 = g(a1, b1)

gr = 2*a*b**3*math.cos(theta) + 3*a**2*b**2*math.sin(theta)

print(a,b, w1, w2, (w2-w1)/h)



import math

r = 3
theta = 3


def x(r, theta):
    return r*math.sin(theta)


def y(r, theta):
    return r*math.cos(theta)

a = x(r, theta)
b = y(r, theta)

print(a,b)

def g(a,b):
    return a**2 * b**3


w = g(a, b)
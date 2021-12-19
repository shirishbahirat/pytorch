import numpy as np


points = [(np.array([4]), 2), (np.array([2]), 4)]

d = 1


def F(w):
    return sum((w.dot(x) - y)**2 for x, y in points)


def dF(w):
    return sum(2 * (w.dot(x) - y) * x for x, y in points)


def gradientDecent(F, dF, d):
    w = np.zeros(d)
    eta = 0.001
    for t in range(100):
        value = F(w)
        gradient = dF(w)

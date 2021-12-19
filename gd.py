import numpy as np

true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []

for i in range(1000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    print(x, y)
    points.append((x, y))


def F(w):
    return sum((w.dot(x) - y)**2 for x, y in points)


def dF(w):
    return sum(2 * (w.dot(x) - y) * x for x, y in points)


def gradientDecent(F, dF, d):
    w = np.zeros(d)
    eta = 0.0001
    for t in range(1000):
        loss = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, loss))


gradientDecent(F, dF, d)

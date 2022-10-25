import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-10,10, 200)
h = 0.001

zi = np.linspace(-4,4, 300)

def expf(x, base):
    return base**x

r = []

for z in zi:
    y = expf(x,z)
    ygrad = (expf(x+h,z) - expf(x,z))/h
    r.append(y[10]/ygrad[10])

plt.plot(zi,r)
plt.grid()
plt.show()


t = []
w = []
theta = np.linspace(2,85, 200)

for th in theta:

    th = math.radians(th)

    s = np.sin(th)
    c = np.cos(th)

    sg = (np.sin(th+h) - np.sin(th))/h
    cg = (np.cos(th+h) - np.cos(th))/h

    t.append(s/sg)
    w.append(c/cg)

    print(s, sg, c, cg)


plt.plot(theta,t)
plt.plot(theta,w)
plt.grid()
plt.show()

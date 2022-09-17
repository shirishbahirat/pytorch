import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

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

h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L


d1 = a*b + c
c += h
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)
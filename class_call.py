import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class model:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self.h = 0.00001
        self._grad = 0.0
        self._op = _op
        self.label = label

    def __call__(self, y):
        print(self.data)
        print('from class call {self.data}')

    def __repr__(self):
        return f"model(data={self.data, self.label})"

    def chain(self):
        print(self._prev)
        return self._prev

    def __add__(self, other):
        out = model(self.data + other.data, (self, other), '+')
        self._grad = 1
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        out = model(self.data * other.data, (self, other), '*')
        self._grad = other.data
        return out

    def backward(self):
        print(f"grad {self._grad}")
        return self._grad

    def subfunction(self):

        def _sub():
            print('from sub ....')

        self.sub = _sub


def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot


def mlp():

    a = model(2.0)
    b = model(3.0)

    c = model(8.0)

    d = a*b + c

    print(d)



def main():

    m = model(6.0)
    m.label = 'm'
    m(10.0)

    m.subfunction()
    m.sub()

    n = model(12.0)
    n.label = 'n'
    n(11.0)

    g = m + n
    g.label = 'g'
    g(2)

    print('b')

    g.chain()

    a = model(2.0)
    a.label = 'a'

    a(1000.0)

    s = g * a
    s.label = 's'

    print('s')

    list(s.chain())[0].chain()

    list(s.chain())[0].backward()

    w = a * g
    w.label = 'w'

    list(w.chain())[1].backward()

    nodes, edges = trace(w)

    print('.....')
    print(nodes)
    print(edges)
    print('.....')

    #y = sum([m, n])

    def test(cc, vv):
        xx = cc

        yy = vv

        print('global objects')

        print(xx, yy)

    test(m, n)

    mlp()

if __name__ == '__main__':

    main()



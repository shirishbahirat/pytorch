
class model:

    def __init__(self, data, _children=()):

        self.data = data

        self._prev = set(_children)
    
        self.h = 0.00001
        self._grad = 0.0

    def __call__(self, y):
        print(self.data)
        print('from class call {self.data}')

    def __repr__(self):
        return f"model(data={self.data})"

    def chain(self):
        print(self._prev)
        return self._prev

    def __add__(self, other):
        out = model(self.data + other.data, (self, other))
        self._grad = 1
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        out = model(self.data * other.data, (self, other))
        self._grad = other.data
        return out

    def backward(self):
        print(f"grad {self._grad}")
        return self._grad

    def subfunction(self):

        def _sub():
            print('from sub ....')

        self.sub = _sub

def main():

    m = model(6.0)
    m(10.0)

    m.subfunction()
    m.sub()

    n = model(12.0)
    n(11.0)

    g = m + n
    g(2)

    print('b')

    g.chain()

    a = model(2.0)

    a(1000.0)

    s = g * a

    print('s')

    list(s.chain())[0].chain()

    list(s.chain())[0].backward()

    w = a * g

    list(w.chain())[1].backward()

    #y = sum([m, n])

    def test(cc, vv):
        xx = cc

        yy = vv

        print('global objects')

        print(xx, yy)

    test(m, n)

if __name__ == '__main__':

    main()



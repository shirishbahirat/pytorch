
def grad(a, b, c):
    return a*b + c

h = 0.0001

f = grad(2,3,-2)
fh = grad(2,3,-2+h)

g = (fh - f)/h

print(g)


def gradn(w1, w2, x1, x2, b):
    return w1*x1 + w1*x2 + w2*x1 + w2*x2 + b


hm = gradn(5,3,10,7,-2)
hd = gradn(5+h,3,10,7,-2)

xc = (hd - hm)/h

print(xc)


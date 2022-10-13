
def grad(a, b, c):
	return a*b + c

h = 0.0001

f = grad(2,3,-2)
fh = grad(2,3+h,-2)

g = (fh - f)/h

print(g)
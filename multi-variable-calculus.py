

t = 2
h = 0.001

def fx(t):
	return t**2

def fy(t):
	return 3*t


def fw(t):
	return fx(t) * fy(t)

print((fw(t+h) - fw(t))/h)
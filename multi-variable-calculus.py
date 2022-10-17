

t = 2
h = 0.00001

def fx(t):
	return t**2

def fy(t):
	return 3*t


def fw(s):
	return fx(s) * fy(s)

print(fw(t+h) - fw(t))

print((fw(t+h) - fw(t))/h)


t = 2
h = 0.000001

def fx(t):
	return t**2

def fy(t):
	return 3*t


def fw(s):
	return fx(s) * fy(s)

print((fw(t+h) - fw(t))/h)


def cal(t):
	return (2*t * 3*t) + (3* t**2)

print(cal(t))


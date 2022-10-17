

t = 2

def fx(t):
	return t**2

def fy(t):
	return 3*t


def fw(t):
	return fx(t) * fy(t)

print(fw(t))
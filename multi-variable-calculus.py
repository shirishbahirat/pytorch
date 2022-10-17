

t = 0.01

def fx(t):
	return t**2

def fy(t):
	return 3*t


def fw():
	return fx(t) * fy(t)

print(fw())
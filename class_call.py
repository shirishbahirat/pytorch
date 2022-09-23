
class model:

	def __init__(self, data, _children=()):
		self.data = data
		self._prev = set(_children)

	def __call__(self, y):
		print(self.data)
		print('from class call {self.data}')

	def __repr__(self):
		return f"Value(data={self.data})"

	def chain(self):
		print(self._prev)

	def __add__(self, other):
		out = model(self.data + other.data, (self, other))
		return out

def main():

	m = model(6.0)
	m(10.0)

	n = model(12.0)
	n(11.0)

	g = m + n
	g(2)

    print('b')
	g.chain()

if __name__ == '__main__':

    main()



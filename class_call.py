

class model:

	def __init__(self, data):
		self.data = data

	def __call__(self, y):
		print(self.data)
		print('from class call {}'.(self.data))


	def __repr__(self):
		print(f'model:')


	def __add__(self, other):
		out = model(self.data + other.data)
		return out




def main():

	m = model(6.0)
	m(10.0)

	n = model(12.0)
	n(11.0)


	g = m + n
	g(2)



if __name__ == '__main__':
    main()



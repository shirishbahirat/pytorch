

class model:

	def __init__(self, data):
		self.data = data

	def __call__(self, y):
		print(self.data)
		print('from class call')


	def __repr__(self):
		print(f'model:')


	def __add__(self, val):
		out = model(self.data + val)
		return out




def main():

	m = model(6)
	m(10)

	n = model(12)
	n(11)


	g = m + n



if __name__ == '__main__':
    main()



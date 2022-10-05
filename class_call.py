

class model:

	def __init__(self, data):
		self.data = data

	def __call__(self, y):
		print(self.data)
		print('from class call')
		for i in self.data:
			print(i + y)

	def __repr__(self):
		print(f'model:')





def main():

	m = model([1,2,3])
	m(10)

	print(m([20]))



if __name__ == '__main__':
    main()



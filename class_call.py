

class model:

	def __init__(self, data):
		self.data = data

	def __call__(self):
		print(self.data)

	def __repr__(self):
		print(f'model:')



def main():

	m = model([1,2,3])
	m()


if __name__ == '__main__':
    main()



class model:

	def __init__(self, data):
		self.data = data

	def __call__(self):
		print(self.data)

	def __repr__(self):
		print(f'model:')

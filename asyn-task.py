import asyncio


class consumer(object):

	def __init__(self):
		self.count = 0
		self.process = asyncio.create_task(self.dispatcher())
		await self.process

	async def dispatcher(self):
		print(f'task {self.count} started')
		await asyncio.sleep(4)
		print(f'task {self.count} completed')
		self.count +=1
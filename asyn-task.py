import asyncio


class consumer(object):

    def __init__(self):
        self.count = 0
        self.limit = 10

    async def scheduler(self):
        self.process = asyncio.create_task(self.dispatcher())
        await self.process

    async def dispatcher(self):

        while True:
            print(f'task {self.count} started')
            await asyncio.sleep(.1)
            print(f'task {self.count} completed')
            self.count +=1
            if self.count > self.limit:
                break


def main():
    cn = consumer()
    asyncio.run(cn.scheduler())

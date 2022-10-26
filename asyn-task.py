import asyncio
from queue import Queue as q

class consumer(object):

    def __init__(self):
        self.count = 0
        self.limit = 10
        asyncio.run(self.scheduler())

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
    


if __name__ == '__main__':

    main()


import asyncio

async def hello(i):
    print(f"hello {i} started")
    await asyncio.sleep(4)
    print(f"hello {i} done")

async def main():
    task1 = asyncio.create_task(hello(1))
    await asyncio.sleep(3)
    task2 = asyncio.create_task(hello(2))
    await task1
    await task2

class test(object):

    def __init__(self, x):
        self.x = x

    @classmethod
    def test(self):
        print (self.x)

asyncio.run(main())  # main loop



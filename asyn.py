import asyncio

async def hello(i):
    print(f"hello {i} started")
    await asyncio.sleep(4)
    print(f"hello {i} done")


class test(object):

    def __init__(self, x):
        self.x = x

    @classmethod
    def test(cls):
        print (self.x)


async def main():

    t = test(10)

    task1 = asyncio.create_task(hello(1))
    await asyncio.sleep(3)
    task2 = asyncio.create_task(hello(2))
    await task1
    await task2


asyncio.run(main())  # main loop



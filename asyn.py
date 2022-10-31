import asyncio

async def hello(i):
    print(f"hello {i} started")
    await asyncio.sleep(4)
    print(f"hello {i} done")


class test(object):

    def __init__(self):
        self.x = 10

    @staticmethod
    def test(b):
        print(100*b)

    @classmethod
    def test(cls, b):
        print (10*b)

    @classmethod
    def another(cls, b):
        print (10*b)

    @staticmethod
    def another(b):
        print(100*b)


async def main():

    test.another(2)
    task1 = asyncio.create_task(hello(1))
    await asyncio.sleep(3)
    task2 = asyncio.create_task(hello(2))
    await task1
    await task2


asyncio.run(main())  # main loop



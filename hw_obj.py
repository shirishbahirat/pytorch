

class alu(object):

    def __init__(self):
        self.value = 0

    def step(self):
        self.value += 1
        yield self.value - 1

    def next(self):
        return next(self.step())


a = alu()

for i in range(10):
    print(a.next())

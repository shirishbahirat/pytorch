

class alu(object):

    def __init__(self):
        self.value = 0
        self.gen = ''

    def step(self):
        while (self.value < 10):
            self.value += 1
            yield self.value - 1

    def instance(self):
        self.gen = self.step()

    def next(self):
        return (next(self.gen))


a = alu()

a.instance()

for i in range(10):
    print(a.next())

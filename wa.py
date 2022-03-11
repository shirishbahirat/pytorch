import simpy
import attr
from queue import Queue as q
from copy import deepcopy as dc
from random import randrange

@attr.s
class cmd(object):
    opcd = attr.ib(0)
    time = attr.ib(0)
    dest = attr.ib(0)
    life = attr.ib(0)

class queue(object):

    def __init__(self, luns, cap, qd, life):
        self.host_queue = [q() for _ in range(luns)]
        self.media_queue = [q() for _ in range(luns)]
        self.luns = luns
        self.cap = cap
        self.qd = qd
        self.life = life

class host(object):

    def __init__(self, env, queue):
        self.host_queue = queue.host_queue
        self.proc = env.process(self.cmd_proc())
        self.qd = queue.qd
        self.life = queue.life
        self.luns = queue.luns
        self.env = env

    def cmd_proc(self):

        while True:
            yield self.env.timeout(1)
            life = randrange(0, self.life)
            dest = randrange(0, self.luns)
            cm = cmd(opcd=1,
                     time=self.env.now,
                     dest = dest,
                     life = life)
            if self.host_queue[dest].qsize() < self.qd:
                self.host_queue[dest].put(dc(cm))

            del life, cm

class ssd(object):

    def __init__(self, env, queue):

        self.env = env
        self.luns = queue.luns
        self.proc = [env.process(self.cmd_proc(id)) for id in range(self.luns)]
        self.read_time = 10
        self.read_time = 20
        self.written = [0 for _ in range(self.luns)]
        self.cap = queue.cap
        self.state = [0 for _ in range(self.luns)]
        self.host_queue = queue.host_queue
        self.media_queue = queue.media_queue

    def cmd_proc(self, id):

        while True:
            yield self.env.timeout(1)
            if self.host_queue[id].qsize():
                if (self.written[id] < self.cap) and (self.state[id] == 0):
                    self.written[id] += 1

                    if self.written[id] == self.cap:
                        self.state[id] = 1

                    cm = self.host_queue[id].queue[0]
                    self.media_queue[id].put(dc(cm))
                    self.host_queue[id].get_nowait()

                    print(id, self.written[id], self.state[id])
                    if cm.opcd:
                        yield self.env.timeout(self.read_time)
                    else:
                        yield self.env.timeout(self.write_time)

                    del cm

    def invalidate(self, id):

        while True:
            pass
            # loop through all cmds in media queue
            # check current time - cmd time
            # if command time above more than cmd life then set cmd life to 0
            # decrement validity



def main():

    env = simpy.Environment()

    q = queue(10, 10, 4, 8)
    h = host(env, q)
    s = ssd(env, q)
    env.run(1000)


if __name__ == "__main__":
    main()

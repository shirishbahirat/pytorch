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
        self.mdia_queue = [q() for _ in range(luns)]
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

    def cmd_proc(self):

        while True:
            yield self.timeout(1)
            life = randrange(0, self.life)
            dest = randrange(0, self.luns)
            cm = cmd(opcd=1,
                     time=self.env.now,
                     dest = dest,
                     life = life)
            if self.host_queue[dest].qsize() < self.qd:
                self.host_queue[dest].put(dc(cmd))

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

    def cmd_proc(self, id):

        while True:
            yield self.env.timeout(1)
            if self.host_queue[id].qsize():
                if (self.written[id] < self.capacity) and (self.state[id] == 0):
                    self.written[id] += 1

                    if self.written[id] = self.cap:
                        self.state[id] = 1

                cmd = self.host_queue[id][0]
                self.mdia_queue[id].put(dc(cmd))
                self.self.host_queue[id].get_nowait()
                del cmd

                if cmd.opcd:
                    yield self.env.timeout(self.read_time)
                else:
                    yield self.env.timeout(self.write_time)


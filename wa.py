import simpy
import attr
from queue import Queue as q
from copy import deepcopy as dc

@attr.s
class cmd(object):
    opcd = attr.ib(0)
    time = attr.ib(0)
    dest = attr.ib(0)
    life = attr.ib(0)

class queue(object):

	def __init__(self, luns):
		self.host_queue = [q() for _ in range(luns)]

class ssd(object):

    def __init__(self, env, luns, cap, queue):

        self.env = env
        self.processor = [env.process(self.cmd_proc(id)) for id in range(luns)]
        self.read_time = 10
        self.written = [0 for _ in range(luns)]
        self.capacity = capacity
        self.state = [0 for _ in range(luns)]
        self.host_queue = queue.host_queue 


    def cmd_proc(self, id):

        while True:
            yield self.env.timeout(1)
            if self.host_queue[id].qsize():
            	if self.written[id] < self.capacity:


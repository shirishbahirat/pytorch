import simpy
from queue import Queue
import attr
from random import randint
from random import expovariate
import time
from copy import deepcopy
from random import seed
import numpy as np


@attr.s
class ipv4_udp(object):
    src = attr.ib('198.196.10.1')
    dst = attr.ib('198.196.10.12')
    zrs = attr.ib(0)
    prt = attr.ib(8800)
    lgh = attr.ib(32)
    spr = attr.ib(1)
    dpr = attr.ib(1)
    lnh = attr.ib(256)
    dat = attr.ib(1024)


@attr.s
class ipv6_udp(object):
    src = attr.ib()
    dst = attr.ib()
    lgh = attr.ib()
    zrs = attr.ib()
    prt = attr.ib()
    spr = attr.ib()
    dpr = attr.ib()
    lnh = attr.ib()
    chk = attr.ib()
    dat = attr.ib()


@attr.s
class tcp(object):
    src = attr.ib()
    dst = attr.ib()
    seq = attr.ib()
    num = attr.ib()
    off = attr.ib()
    nss = attr.ib()
    cwr = attr.ib()
    ece = attr.ib()
    urg = attr.ib()
    ack = attr.ib()
    psh = attr.ib()
    rst = attr.ib()
    syn = attr.ib()
    fin = attr.ib()
    win = attr.ib()
    chk = attr.ib()
    urp = attr.ib()
    opt = attr.ib()


class generator(object):

    def __init__(self, env, id=0, rate=1.0, data=0.01, init_delay=1.0, packets=1e6, flow_id=0):
        self.id = id
        self.env = env
        self.rate = rate
        self.data = data
        self.init_delay = init_delay
        self.packets = packets
        self.out = Queue()
        self.packets_sent = 0
        self.action = env.process(self.dispatch())
        self.flow_id = flow_id

    def arrival(self):
        return self.rate

    def distSize(self):
        return expovariate(self.data)

    def dispatch(self):
        yield self.env.timeout(self.init_delay)
        while self.packets_sent < self.packets:
            yield self.env.timeout(self.arrival())
            self.packets_sent += 1
            print(self.env.now, self.packets_sent)
            p = ipv4_udp()
            self.out.put(p)


def main():
    env = simpy.Environment()
    h = generator(env)

    env.run(until=10000)


if __name__ == '__main__':
    main()

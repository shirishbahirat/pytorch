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
    src = attr.ib()
    dst = attr.ib()
    zrs = attr.ib()
    prt = attr.ib()
    lgh = attr.ib()
    spr = attr.ib()
    dpr = attr.ib()
    lnh = attr.ib()
    dat = attr.ib()


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

    def __init__(self, env, id, adist, sdist, initial_delay=0, finish=float("inf"), flow_id=0):
    self.id = id
    self.env = env
    self.adist = adist
    self.sdist = sdist
    self.initial_delay = initial_delay
    self.finish = finish
    self.out = None
    self.packets_sent = 0
    self.action = env.process(self.run())  # starts the run() method as a SimPy process
    self.flow_id = flow_id

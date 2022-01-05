import simpy
from queue import Queue
import attr
from random import randint
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

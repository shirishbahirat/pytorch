import torch
import torch.nn as nn
from random import random 

x = torch.tensor([[float(i)*.1] for i in range(10)])
y = x*-3.0 + 3.3 + random()


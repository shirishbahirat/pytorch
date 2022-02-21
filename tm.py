import torch
import torch.nn as nn
import torch.optim as optim
from random import random 

x = torch.tensor([[i*0.1] for i in range(10)])
y = torch.tensor([[i*-3.3 + 10 + random()] for i in range(10)])

print(x.T,y.T)


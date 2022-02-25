import re
from turtle import forward
import torch
import torch.nn as nn

import torch.nn.functional as F
from urllib3 import Retry

data = torch.FloatTensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.2], [0.2, 0.2, 0.6]])
label = torch.FloatTensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.1]])


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.l1 = nn.Linear(3,6)

    def forward(self,x):
        y = self.l1(x)
        return y


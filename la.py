import torch
import torch.nn as nn
from random import random
import torch.optim as optim


data = torch.tensor([[.101]])
label = torch.tensor([[1]]) 

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(x)

    def forward(self,x):
        y = self.linear(x)
        y = F.relu(y)
        return y

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim(model.parameters(), lr=0.001,momentum=0.9)

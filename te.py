import torch
import torch.nn as nn
from random import random 

x = torch.tensor([[float(i)*.1] for i in range(10)])
y = x*-3.0 + 3.3 + random()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.liner = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.liner(x)
        return y_pred

model = Model()

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


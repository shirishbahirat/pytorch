import torch
import torch.nn as nn
from random import random
import torch.optim as optim
import torch.nn.functional as F


data = torch.tensor([[.101]])
label = torch.tensor([[.1]]) 


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        y = self.linear(x)
        y = F.relu(y)
        return y

model = Model()

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)


for epoch in range(1000):

    output = model(data)

    loss = criterion(output, label)

    print('Epoch {:4.0f} | Loss {:4.3f}'.format(epoch, loss.item()))
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()


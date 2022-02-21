from statistics import mode
import torch
import torch.nn as nn
import torch.optim as optim
from random import random 

x = torch.tensor([[i*0.1] for i in range(10)])
y = torch.tensor([[i*-1.3 + 2 + random()] for i in range(10)])

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

print(model)

criteria = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x)
    loss = criteria(y_pred, y)
    print('Epoch {:4.0f} | Loss {:4.5f}'.format(epoch, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(x.T,y.T)

p = 0.9
x = torch.tensor([[p]])
y_pred = model(x)
print('Predicted {:2.2} : {:4.5}'.format(p,y_pred.data[0][0].item()))

from statistics import mode
import torch
import torch.nn as nn
from random import random

x = torch.tensor([[float(i)*.1] for i in range(10)])
y = x*2. + 2. + random()

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = Model()
print(model)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    print('Epoch {:4.0f} | Loss {:3.4f}'.format(epoch, loss.item()))
    
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()


p = 5.
tx = torch.tensor([[p]])
y_pred = model(tx)

print('Predict {:3.3f} : {:4.4f}'.format(p, y_pred.data[0][0].item()))

from statistics import mode
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

for epoch in range(500):

    y_pred = model(x)
    loss = criterion(y_pred, y)

    print('Epoch {:4.0f} | Loss {:4.4f}'.format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

p = .1
x_data = torch.tensor([[p]])
y_pred = model(x_data)

print('Predicted {} : {}'.format(p, y_pred[0][0].item()))
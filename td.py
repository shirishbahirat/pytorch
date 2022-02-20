import torch
import torch.nn as nn
from random import random

x = torch.tensor([[float(i)] for i in range(5)])
y = x*2. + 3 + random()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.l1(x)
        return y_pred

model = Model()

criteria = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


for epoch in range(100):
    y_pred = model(x)
    loss = criteria(y_pred, x)

    print('Epoch {:3.0f} loss {:4.4f}'.format(epoch,
    loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


p = 5.0
y_hat = model(torch.tensor([[p]]))
print(x, y, y_hat)

print('Prediction {}:{}'.format(p,y_hat.data[0][0].item()))

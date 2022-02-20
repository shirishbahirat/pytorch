import torch
import torch.nn as nn
from random import random

x = torch.tensor([[float(i)*.1] for i in range(10)])
y = x*2. + 3 + random()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1,1)

    def forward(self,x):
        return self.l1(x)

model = Model()

criteria = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


for epoch in range(1000):
    y_pred = model(x)
    loss = criteria(y_pred, x)

    print('Epoch {:3.0f} loss {:4.4f}'.format(epoch,
    loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


y_hat = model(torch.tensor([[.09]]))
print(x, y, y_hat)

print('Prediction {}:{}'.format(0.09,y_hat.data[0][0].item()))

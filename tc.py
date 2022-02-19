import torch
import torch.nn as nn
from random import random 

X = torch.tensor([[float(i)] for i in range(5)])
Y = X*3.5 + random()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

print(model)

for epoch in range(200):
    y_pred = model(X)

    loss = criterion(y_pred, Y)
    print(f'Epoch: {epoch:3.0f} | Loss {loss.item():.3f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


p = 10.
tx = torch.tensor([[p]])
y_hat = model(tx)

print('Predection after training {} {:.2f}'.format(p, y_hat.data[0][0].item()))


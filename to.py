import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

a = np.array([(i*i) for i in range(-9,10)])
b = np.array([(i*i*i+5.0) for i in range(-9,10)])

a = a/max(a)
b = b/max(b)

input = torch.tensor(a, dtype=torch.float32, requires_grad=True)
target = torch.tensor(b, dtype=torch.float32)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(19,32)
        self.linear2 = nn.Linear(32,128)
        self.linear3 = nn.Linear(128,64)
        self.linear4 = nn.Linear(64,19)

    def forward(self,x):
        y_hat = F.relu(self.linear1(x))
        y_hat = F.relu(self.linear2(y_hat))
        y_hat = F.relu(self.linear3(y_hat))
        y_hat = F.relu(self.linear4(y_hat))
        return y_hat

model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loss = []

for epoch in range(1000):

    y_hat = model(input)
    optimizer.zero_grad()
    loss = criterion(y_hat, target)
    train_loss.append(loss)
    print('Epoch {:4.0f} | Loss {:4.4f}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()


out= model(input)
print(out, a, b)




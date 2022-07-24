import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

a = [i*i for i in range(-9,10)]
b = [i*i+5 for i in range(-9,10)]

input = torch.tensor(a, dtype=float, requires_grad=True)
target = torch.tensor(b, dtype=float, requires_grad=True)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(19,10)
        self.linear2 = nn.Linear(10,19)
        self.linear3 = nn.Linear(8,19)

    def forward(self,x):
        y_hat = F.relu(self.linear1(x))
        y_hat = F.relu(self.linear2(y_hat))
        y_hat = F.relu(self.linear3(y_hat))
        return y_hat

model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_loss = []

for epoch in range(1000):

    y_hat = model(input)
    optimizer.zero_grad()
    loss = criterion(y_hat, target)
    train_loss.append(loss)
    print('Epoch {:4.0f} | Loss {:4.4f}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()


print(input)
print(target)



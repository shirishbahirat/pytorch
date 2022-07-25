import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

a = np.array([(i*i) for i in range(-9,10)])
b = np.array([(i*i+5.0) for i in range(-9,10)])



a = a/max(a)
b = b/max(b)

a = np.array([0.073096932, 0.261599879, 0.414636285, 0.53391308, 0.621649691, 0.680578049, 0.713942582, 0.725500221, 0.719520394, 0.700785031, 0.674588563, 0.64673792, 0.62355253, 0.611864326, 0.619017737, 0.652869693, 0.721789626, 0.834659466, 1.000873644])

input = torch.tensor(a, dtype=torch.float32, requires_grad=True)
target = torch.tensor(b, dtype=torch.float32)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(19,22)
        self.linear2 = nn.Linear(22,32)
        self.linear3 = nn.Linear(32,22)
        self.linear4 = nn.Linear(22,19)

    def forward(self,x):
        y_hat = F.relu(self.linear1(x))
        y_hat = F.relu(self.linear2(y_hat))
        y_hat = F.relu(self.linear3(y_hat))
        y_hat = F.sigmoid(self.linear4(y_hat))
        return y_hat

model = Model()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_loss = []

for epoch in range(5000):

    y_hat = model(input)
    optimizer.zero_grad()
    loss = criterion(y_hat, target)
    train_loss.append(loss)
    print('Epoch {:4.0f} | Loss {:4.4f}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()


out= model(input)
print(out, a, b)

plt.plot(out.detach().numpy())
plt.plot(target.detach().numpy()+0.009)
plt.plot(input.detach().numpy())
plt.show()



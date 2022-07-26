import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

a1 = np.array([(i*i) for i in range(-9,10)])
b1 = np.array([(i*i+5.0) for i in range(-9,10)])

a1 = a1/max(a1)
b1 = b1/max(b1)

a1 = np.array([0.073096932, 0.261599879, 0.414636285, 0.53391308, 0.621649691, 0.680578049, 0.713942582, 0.725500221, 0.719520394, 0.700785031, 0.674588563, 0.64673792, 0.62355253, 0.611864326, 0.619017737, 0.652869693, 0.721789626, 0.834659466, 1.000873644])
a2 = np.array([0.557912174, 0.583314627, 0.608367332, 0.631376756, 0.650451583, 0.663502717, 0.668243283, 0.662188623, 0.642656299, 0.606766091, 0.55144, 0.473402244, 0.369179263, 0.235099712, 0.06729447, -0.13830337, -0.38595849, -0.680133359, -1.025488223])

input1 = torch.tensor(a1, dtype=torch.float32, requires_grad=True)
input2 = torch.tensor(a2, dtype=torch.float32, requires_grad=True)
target = torch.tensor(b1, dtype=torch.float32)


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

model1 = Model()
model2 = Model()

criterion1 = nn.MSELoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)

criterion2 = nn.MSELoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

train_loss1 = []
train_loss2 = []

for epoch in range(5000):

    y_hat1 = model1(input1)
    y_hat2 = model2(input2)

    optimizer1.zero_grad()
    optimizer2.zero_grad()

    loss1 = criterion1(y_hat1, target)
    loss2 = criterion2(y_hat2, target)

    train_loss1.append(loss1)
    train_loss2.append(loss2)

    print('Epoch {:4.0f} | Loss {:4.4f, 4.4f}'.format(epoch, loss1.item(), loss2.item()))

    loss1.backward()
    optimizer1.step()

    loss2.backward()
    optimizer2.step()

out1 = model1(input1)
out2 = model2(input2)

plt.plot(out1.detach().numpy())
plt.plot(out2.detach().numpy())
plt.plot(target.detach().numpy())
plt.plot(input1.detach().numpy())
plt.plot(input2.detach().numpy())
plt.show()

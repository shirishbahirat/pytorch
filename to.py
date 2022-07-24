import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

input = torch.randn(10,8, requires_grad=True)
target = torch.randn(10, 5).softmax(dim=1)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8,10)
        self.linear2 = nn.Linear(10,8)
        self.linear3 = nn.Linear(8,5)

    def forward(self,x):
        y_hat = F.relu(self.linear1(x))
        y_hat = F.relu(self.linear2(y_hat))
        y_hat = F.relu(self.linear3(y_hat))
        return y_hat

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_loss = []

for epoch in range(200):

    y_hat = model(input)
    optimizer.zero_grad()
    loss = criterion(y_hat, target)
    train_loss.append(loss)
    print('Epoch {:4.0f} | Loss {:4.4f}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()

'''
plt.plot(train_loss)
plt.show()
'''


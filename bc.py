import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

data = torch.FloatTensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.2], [0.2, 0.2, 0.6]])
labels = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.l1 = nn.Linear(3,6)
        self.l2 = nn.Linear(6,3)

    def forward(self,x):
        y = self.l1(x)
        y = self.l2(y)
        return y[:, -1,:]


module = Net()



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(module.parameters(), lr=0.01)

for epoch in range(10):

    y_hat = module(data)

    loss = criterion(y_hat[0], labels[0])


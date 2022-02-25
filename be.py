import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

data = torch.FloatTensor([[[1.8, 0.1, 0.1]],
            [[0.2, 0.1, 3.1]]])
labels = torch.LongTensor([[0],[2]])



class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()

        self.l1 = nn.Linear(3,6)
        self.l2 = nn.Linear(6,3)

    def forward(self,x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y


module = Net()



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(module.parameters(), lr=0.001)

for epoch in range(1000):

    y_hat = module(data)

    loss = criterion(y_hat, labels)

    print('Epoch {:4.0f} : Loss {:2.4f}'.format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(module(torch.FloatTensor([0.1,0.2, 1.9]))[0].data.item())
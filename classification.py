import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.F.relu(self.conv1(x)))
        x = self.pool(torch.F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.F.relu(self.fc1(x))
        x = torch.F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for each in range(2):

    running_loss = 0.00

    for i, data in enumerate(trainloader, 0):

        inputs, lables = data

        inputs, lables = Variable(inputs), Variable(lables)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, lables)

        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if i % 2000 == 1999:
            print('[%d, %5d] loss: %,3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


print('finished training')

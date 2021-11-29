import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


x_data = Variable(torch.tensor([[1.], [2.], [3.]]))
y_data = Variable(torch.tensor([[2.], [4.], [6.]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model()


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):

    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hr = Variable(torch.tensor([4.]))
model.forward(hr)

print ("predict (after training)", 4, model.forward(hr).item())

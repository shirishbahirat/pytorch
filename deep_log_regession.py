import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)

x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


model = Model()


criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


hour_var = Variable(torch.tensor([[1.0]]))
print("predict (after training)", 1, model.forward(hour_var).data[0][0])
hour_var = Variable(torch.tensor([[7.0]]))
print("predict (after training)", 7, model.forward(hour_var).data[0][0])

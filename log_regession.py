import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data = Variable(torch.tensor([[1.], [2.], [3.], [4.]]))
y_data = Variable(torch.tensor([[0.], [0.], [1.], [1.]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = Model()


criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


hour_var = Variable(torch.tensor([[4.0]]))
print("predict (after training)", 1, model.forward(hour_var).data[0][0])
print("predict (after training)", 7, model.forward(hour_var).data[0][0])

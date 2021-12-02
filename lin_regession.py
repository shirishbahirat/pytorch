import torch
from torch.autograd import Variable

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


criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


hour_var = Variable(torch.tensor([[4.0]]))
print("predict (after training)", 4, model.forward(hour_var).data[0][0])

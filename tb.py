import torch
import torch.nn as nn
from torch.autograd import Variable

X = Variable(torch.tensor([[1.],[2.],[3.],[4.],[5.]]))
Y = Variable(torch.tensor([[2.],[4.],[6.],[8.],[10.]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model()

print(model)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    y_pred = model(X)

    loss = criterion(y_pred, Y)
    print(f'Epoch: {epoch:3.0f} | Loss: {loss.item():.3f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

para = 10.
tx = torch.tensor([[para]])
y_hat = model(tx)

print('Predection after training {} {:.2f}'.format(para, y_hat.data[0][0].item()))





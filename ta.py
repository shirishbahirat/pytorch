from statistics import mode
import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(torch.tensor([[1.],[2.],[3.],[4.],[5.]]))
y = Variable(torch.tensor([[2.],[4.],[6.],[8.],[10.]]))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)
    
    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(f'Epoch: {epoch:3.0f} | Loss: {loss.item():.3f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

para = 20.
tv = torch.tensor([[para]])
y_hat = model(tv)

print('Predection after training {} {:.2f}'.format(para, y_hat.data[0][0].item()))

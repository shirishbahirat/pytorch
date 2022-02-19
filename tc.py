import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable

X = torch.tensor([[i] for i in range(10)])
Y = X*2

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

print(model)

for epoch in range(100):
    y_pred = model(x)

    loss = criterion(y_pred, Y)

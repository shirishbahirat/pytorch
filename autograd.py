import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

x = torch.randn(10, 3)
y = torch.randn(10, 2)

print(x)
print(y)

linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())

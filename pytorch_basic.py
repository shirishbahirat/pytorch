import torch
import torchvision
from torch.autograd import Variable
import numpy as np


def f(x):
    return x ** 2 + 2 * x + 6


np_x = np.array([4.0])
x = torch.from_numpy(np_x).requires_grad_(True)
y = f(x)
print(y)

y.backward()
x.grad
print(x.grad)

np_x = np.array([5.0])
x = torch.from_numpy(np_x).requires_grad_(True)
y = f(x)
print(y)

y.backward()
x.grad
print(x.grad)


np_x = np.array([10.])
np_x = np_x.transpose()
x = torch.from_numpy(np_x).requires_grad_(True)
y = f(x)
print(y)

y.backward()
x.grad
print(x.grad)


a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

print(a, b)

Q = 3 * a**3 - b**2
print('Q', Q)
external_grad = torch.tensor([1., 1.])

Q.backward(gradient=external_grad)

print('a grad', a.grad)
print('b grad', b.grad)


x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b

model = torch.nn.Linear(5, 3)


loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print('before',w)

for epoch in range(100):
    z = torch.matmul(x, w) + b
    outputs = model(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, loss.item()))

print('after',w)


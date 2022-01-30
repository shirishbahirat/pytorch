import torch

x = torch.tensor([.1, .2, .3], requires_grad=True)

y = x**2 + 2
print(y)
v = torch.tensor([0.1, 0.01, 0.01], dtype=torch.float32)
y.backward(x)

print(x.grad)

x = torch.randn(3, requires_grad=True)
y = x + 2
y.backward(x)

k = torch.tensor(10., requires_grad=True)
m = k + 10
m.backward()
print(k.grad)

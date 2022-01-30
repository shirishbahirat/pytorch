import torch

x = torch.tensor([.1, .2, .3], requires_grad=True)

y = x**2 + 2
v = torch.tensor([0.1, 0.01, 0.01], dtype=torch.float32)
y.backward(x)

print(x.grad)

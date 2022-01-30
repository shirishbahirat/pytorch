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


'''
x = a(x)
x = b(y)
dx/dx = dz/dy * dy/dx

'''


x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

y_hat = w * x
loss = (y_hat - y)**2

print(loss)

loss.backward()
print(w.grad)

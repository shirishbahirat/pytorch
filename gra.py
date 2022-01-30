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
y = torch.tensor(4.0)

w = torch.tensor(.02, requires_grad=True)

y_hat = w * x + .5
loss = (y_hat - y)**2

print(loss)

loss.backward()
print(w.grad)

for epoch in range(1000):

    y_hat = w * x + .5
    loss = (y_hat - y)**2
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
    w.grad.zero_()

    print(y_hat)

import torch

t1 = torch.tensor(4.)

t2 = torch.tensor([1., 2, 3, 4])

t3 = torch.tensor([[5., 6],
                   [7, 8],
                   [9, 10]])

t4 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[15, 16, 17],
     [17, 18, 19.]]])

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

y = w * x + b
y.backward()

print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)

t6 = torch.full((3, 2), 42)

t7 = torch.cat((t3, t6))

t8 = torch.sin(t7)

t9 = t8.reshape(3, 2, 2)

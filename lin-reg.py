import numpy as np
import torch

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)


def model(x):
    return x @ w.t() + b


preds = model(inputs)


def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


loss = mse(preds, targets)

loss.backward()

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5

loss = mse(preds, targets)


for epoch in range(60):

    outputs = model(inputs)
    loss = mse(outputs, targets)

    w.grad.zero_()
    b.grad.zero_()

    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5

    if (epoch + 1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

'''
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()
'''

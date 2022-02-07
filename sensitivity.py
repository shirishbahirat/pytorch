import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random


a = np.array([[i / 50] for i in range(100)])
a = torch.from_numpy(a.astype(np.float32))

x1 = np.array([[(random.randint(0, 10) * 2 + i * -2) / 50] for i in range(100)])
X1 = torch.from_numpy(x1.astype(np.float32))

x2 = np.array([[(random.randint(0, 10) * 2 + i * 2) / 50] for i in range(100)])
X2 = torch.from_numpy(x2.astype(np.float32))

x3 = np.array([[(random.randint(0, 10) * 2 + i * 4) / 50] for i in range(100)])
X3 = torch.from_numpy(x3.astype(np.float32))

x4 = np.array([[(random.randint(0, 10) * 2 + i * 6) / 50] for i in range(100)])
X4 = torch.from_numpy(x4.astype(np.float32))


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer1(x)


class sensitivity(object):
    def __init__(self, input_data, output_data, input_dim, output_dim):
        self.input_data = input_data
        self.input_dim = input_dim
        self.output_data = output_data
        self.output_dim = output_dim
        self.model = model = Model(self.input_dim, self.output_dim)
        self.learning_rate = 0.02
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        self.num_epochs = 2000
        self.x_predicted = self.output_data
        self.criterion = nn.MSELoss()
        self.loss = self.criterion(self.x_predicted, self.output_data)

    def train(self):

        for epoch in range(self.num_epochs):
            # Forward pass and loss
            self.x_predicted = self.model.forward(self.input_data)
            self.loss = self.criterion(self.x_predicted, self.output_data)

            # Backward pass and update
            self.loss.backward()
            self.optimizer.step()

            # zero grad before new step
            self.optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f'epoch: {epoch+1}, loss = {self.loss.item():.4f}')

    def predict(self):
        return self.model(self.input_data).detach().numpy()

    def get_weight(self):
        weight = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name == 'layer1.weight':
                    weight = param.data

        return weight


sen1 = sensitivity(a, X1, 1, 1)
sen1.train()
predicted = sen1.predict()
print(sen1.get_weight())
w1 = sen1.get_weight()

plt.plot(a, X1, 'ro')
plt.plot(a, predicted, 'b')
plt.show()


sen2 = sensitivity(a, X2, 1, 1)
sen2.train()
predicted = sen2.predict()
print(sen2.get_weight())
w2 = sen2.get_weight()

plt.plot(a, X2, 'ro')
plt.plot(a, predicted, 'b')
plt.show()


sen3 = sensitivity(a, X3, 1, 1)
sen3.train()
predicted = sen3.predict()
print(sen3.get_weight())
w3 = sen3.get_weight()


plt.plot(a, X3, 'ro')
plt.plot(a, predicted, 'b')
plt.show()


sen4 = sensitivity(a, X4, 1, 1)
sen4.train()
predicted = sen4.predict()
print(sen4.get_weight())
w4 = sen4.get_weight()


plt.plot(a, X4, 'ro')
plt.plot(a, predicted, 'b')
plt.show()


w = torch.tensor([w1, w2, w3, w4])
s = torch.einsum('i,j->ij', w, w)
print(s)

s = F.softmax(s, dim=1)
print(s)

import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer1(x)


class atn(object):
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
